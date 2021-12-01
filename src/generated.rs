//! Everything in this module is pending to be refactored, turned into idiomatic-rust, and moved to
//! other modules.

use crate::vector::Vec3;
use core::{cmp::Ordering, ptr::null_mut};

use crate::{face_vert_to_index, get_normal, get_position, get_tex_coord, Geometry};

#[derive(Copy, Clone)]
pub struct TSpace {
    pub os: Vec3,
    pub mag_s: f32,
    pub ot: Vec3,
    pub mag_t: f32,

    pub counter: usize,
    pub orient: bool,
}

impl TSpace {
    pub fn zero() -> Self {
        Self {
            os: Vec3::zero(),
            mag_s: 0.0,
            ot: Vec3::zero(),
            mag_t: 0.0,
            counter: 0,
            orient: false,
        }
    }
}

// To avoid visual errors (distortions/unwanted hard edges in lighting), when using sampled normal maps, the
// normal map sampler must use the exact inverse of the pixel shader transformation.
// The most efficient transformation we can possibly do in the pixel shader is
// achieved by using, directly, the "unnormalized" interpolated tangent, bitangent and vertex normal: vT, vB and vN.
// pixel shader (fast transform out)
// vNout = normalize( vNt.x * vT + vNt.y * vB + vNt.z * vN );
// where vNt is the tangent space normal. The normal map sampler must likewise use the
// interpolated and "unnormalized" tangent, bitangent and vertex normal to be compliant with the pixel shader.
// sampler does (exact inverse of pixel shader):
// float3 row0 = cross(vB, vN);
// float3 row1 = cross(vN, vT);
// float3 row2 = cross(vT, vB);
// float fSign = dot(vT, row0)<0 ? -1 : 1;
// vNt = normalize( fSign * float3(dot(vNout,row0), dot(vNout,row1), dot(vNout,row2)) );
// where vNout is the sampled normal in some chosen 3D space.
//
// Should you choose to reconstruct the bitangent in the pixel shader instead
// of the vertex shader, as explained earlier, then be sure to do this in the normal map sampler also.
// Finally, beware of quad triangulations. If the normal map sampler doesn't use the same triangulation of
// quads as your renderer then problems will occur since the interpolated tangent spaces will differ
// eventhough the vertex level tangent spaces match. This can be solved either by triangulating before
// sampling/exporting or by using the order-independent choice of diagonal for splitting quads suggested earlier.
// However, this must be used both by the sampler and your tools/rendering pipeline.
// internal structure

#[derive(Clone)]
pub struct Triangle {
    pub face_neighbors: [Option<usize>; 3],
    pub assigned_group: [*mut Group; 3],

    pub os: Vec3,
    pub mag_s: f32,
    pub ot: Vec3,
    pub mag_t: f32,

    pub original_face: usize,
    pub tspaces_offset: usize,

    pub flag: u32,
    pub vert_num: [u8; 3],
}

impl Triangle {
    fn zero() -> Self {
        Self {
            face_neighbors: [None; 3],
            assigned_group: [null_mut(), null_mut(), null_mut()],
            os: Vec3::zero(),
            mag_s: 0.0,
            ot: Vec3::zero(),
            mag_t: 0.0,
            original_face: 0,
            flag: 0,
            tspaces_offset: 0,
            vert_num: [0, 0, 0],
        }
    }
}

#[derive(Clone)]
pub struct Group {
    pub start: usize,
    pub end: usize,
    pub vertex_representitive: usize,
    pub orient_preservering: bool,
}

impl Group {
    fn zero() -> Self {
        Self {
            start: 0,
            end: 0,
            vertex_representitive: 0,
            orient_preservering: false,
        }
    }

    fn iter<'a>(&self, buffer: &'a [usize]) -> impl Iterator<Item = usize> + 'a {
        buffer[self.start..self.end].iter().copied()
    }
}

#[derive(Clone)]
pub struct SubGroup {
    pub members: Vec<usize>,
}

impl SubGroup {
    const fn zero() -> Self {
        Self {
            members: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct Edge {
    pub i0: usize,
    pub i1: usize,
    pub f: usize,
}

impl Edge {
    const fn zero() -> Self {
        Self { i0: 0, i1: 0, f: 0 }
    }

    #[inline]
    fn array(&self, channel: usize) -> usize {
        match channel {
            0 => self.i0,
            1 => self.i1,
            2 => self.f,
            _ => unreachable!("wtf?"),
        }
    }
}

#[derive(Clone)]
pub struct TmpVert {
    pub vert: [f32; 3],
    pub index: usize,
}

impl TmpVert {
    fn zero() -> Self {
        Self {
            vert: [0.0, 0.0, 0.0],
            index: 0,
        }
    }
}

pub fn generate_tangent_space<I: Geometry>(geometry: &mut I, angular_threshold: f32) -> bool {
    let mut total_triangles = 0;
    let num_faces = geometry.num_faces();
    let thres_cos = angular_threshold.to_radians().cos();
    for face in 0..num_faces {
        let verts = geometry.num_vertices_of_face(face);
        if verts == 3 {
            total_triangles += 1
        } else if verts == 4 {
            total_triangles += 2
        }
    }

    if total_triangles == 0 {
        return false;
    }

    let mut indices = vec![0usize; 3 * total_triangles];
    let mut triangles = vec![Triangle::zero(); total_triangles];

    let num_tspaces = generate_initial_vertices_index_list(&mut triangles, &mut indices, geometry);
    generate_shared_vertices_index_list(&mut indices, geometry);

    let mut degen_triangles = 0;
    for t in 0..total_triangles {
        let p0 = get_position(geometry, indices[t * 3]);
        let p1 = get_position(geometry, indices[t * 3 + 1]);
        let p2 = get_position(geometry, indices[t * 3 + 2]);
        if p0 == p1 || p0 == p2 || p1 == p2 {
            triangles[t].flag |= 1;
            degen_triangles += 1
        }
    }

    let active_triangles = total_triangles - degen_triangles;
    degen_prologue(&mut triangles, &mut indices, active_triangles);
    init_tri_info(&mut triangles[..active_triangles], &indices, geometry);
    let max_groups = active_triangles * 3;

    let mut groups = vec![Group::zero(); max_groups];
    let mut group_triangle_buffer = vec![0; active_triangles * 3];

    let active_groups = build4_rule_groups(
        &mut triangles[..active_triangles],
        &mut groups,
        &mut group_triangle_buffer,
        &indices,
    );

    let mut tspaces = vec![
        TSpace {
            os: Vec3::new(1.0, 0.0, 0.0),
            ot: Vec3::new(0.0, 1.0, 0.0),
            mag_s: 1.0,
            mag_t: 1.0,
            ..TSpace::zero()
        };
        num_tspaces
    ];

    if !generate_tspaces(
        &mut tspaces,
        &triangles[..active_triangles],
        &groups[..active_groups],
        &indices,
        thres_cos,
        &group_triangle_buffer,
        geometry,
    ) {
        return false;
    }

    degen_epilogue(
        &mut tspaces,
        &triangles,
        &indices,
        geometry,
        active_triangles,
    );

    let mut index = 0;
    for face in 0..num_faces {
        let num_vertices = geometry.num_vertices_of_face(face);
        if !(num_vertices != 3 && num_vertices != 4) {
            for vertex in 0..num_vertices {
                let tspace = &tspaces[index];
                let tangent = Vec3::new(tspace.os.x, tspace.os.y, tspace.os.z);
                let bitangent = Vec3::new(tspace.ot.x, tspace.ot.y, tspace.ot.z);
                geometry.set_tangent(
                    tangent.into(),
                    bitangent.into(),
                    [tspace.mag_s, tspace.mag_t],
                    tspace.orient,
                    face,
                    vertex,
                );
                index += 1;
            }
        }
    }

    true
}

fn degen_epilogue<I: Geometry>(
    tspaces: &mut [TSpace],
    triangles: &[Triangle],
    indices: &[usize],
    geometry: &mut I,
    input_triangles: usize,
) {
    for t in input_triangles..triangles.len() {
        if triangles[t].flag & 2 != 0 {
            continue;
        }

        for i in 0..3 {
            let index = indices[t * 3 + i];

            let position = indices[..3 * input_triangles]
                .iter()
                .position(|&idx| idx == index);

            if let Some(pos) = position {
                let (tri, vert) = (pos / 3, pos % 3);
                let src = triangles[tri].tspaces_offset + triangles[tri].vert_num[vert] as usize;
                let dst = triangles[t].tspaces_offset + triangles[t].vert_num[i] as usize;
                tspaces[dst] = tspaces[src];
            }
        }
    }

    for triangle in &triangles[0..input_triangles] {
        if triangle.flag & 2 != 0 {
            let vtx = triangle.vert_num;
            let flag = 1 << vtx[0] | 1 << vtx[1] | 1 << vtx[2];

            let missing_index = if flag & 2 == 0 {
                1
            } else if flag & 4 == 0 {
                2
            } else if flag & 8 == 0 {
                3
            } else {
                0
            };

            let face_num = triangle.original_face;
            let dst = get_position(geometry, face_vert_to_index(face_num, missing_index));

            for &vertex in &vtx {
                if dst == get_position(geometry, face_vert_to_index(face_num, vertex as usize)) {
                    let offset = triangle.tspaces_offset;
                    tspaces[offset + missing_index] = tspaces[offset + vertex as usize];
                    break;
                }
            }
        }
    }
}

fn generate_tspaces<I: Geometry>(
    tspaces: &mut [TSpace],
    triangles: &[Triangle],
    groups: &[Group],
    indices: &[usize],
    thres_cos: f32,
    group_triange_buffer: &[usize],
    geometry: &mut I,
) -> bool {
    let max_faces = groups.iter().map(|g| g.end - g.start).max().unwrap_or(0);
    if max_faces == 0 {
        return true;
    }

    let mut subgroup_tspace = vec![TSpace::zero(); max_faces];
    let mut uni_subgroups = vec![SubGroup::zero(); max_faces];
    let mut members = Vec::with_capacity(max_faces);

    let mut _unique_tspaces = 0;
    for group in groups {
        let mut unique_subgroups = 0;
        for face_index in group.iter(group_triange_buffer) {
            let face = &triangles[face_index];
            let mut tmp_group = SubGroup {
                members: Vec::new(),
            };

            let group_ptr = group as *const Group as *mut Group;
            let index = if face.assigned_group[0] == group_ptr {
                0
            } else if face.assigned_group[1] == group_ptr {
                1
            } else if face.assigned_group[2] == group_ptr {
                2
            } else {
                continue;
            };

            let vert_index = indices[face_index * 3 + index];
            let n = get_normal(geometry, vert_index);

            let os = (face.os - n.dot(face.os) * n).safe_normalize();
            let ot = (face.ot - n.dot(face.ot) * n).safe_normalize();

            let of1 = face.original_face;
            members.clear();
            for t in group.iter(group_triange_buffer) {
                let of2 = triangles[t].original_face;

                let os2 = (triangles[t].os - n.dot(triangles[t].os) * n).safe_normalize();
                let ot2 = (triangles[t].ot - n.dot(triangles[t].ot) * n).safe_normalize();

                let any = (face.flag | triangles[t].flag) & 4 != 0;
                let same_original_face = of1 == of2;

                let cos_s = os.dot(os2);
                let cos_t = ot.dot(ot2);

                if any || same_original_face || cos_s > thres_cos && cos_t > thres_cos {
                    members.push(t)
                }
            }

            if members.len() > 1 {
                members.sort_unstable();
            }

            tmp_group.members = members.clone();

            let found = uni_subgroups[0..unique_subgroups]
                .iter()
                .position(|b| tmp_group.members == b.members);

            let found = if let Some(found) = found {
                found
            } else {
                uni_subgroups[unique_subgroups].members = tmp_group.members.clone();
                subgroup_tspace[unique_subgroups] = eval_tspace(
                    &tmp_group.members,
                    indices,
                    triangles,
                    geometry,
                    group.vertex_representitive,
                );
                unique_subgroups += 1;
                unique_subgroups - 1
            };

            let offset = face.tspaces_offset;
            let vertex = face.vert_num[index] as usize;
            let mut ts = &mut tspaces[offset + vertex];
            if ts.counter == 1 {
                *ts = average_tspace(ts, &subgroup_tspace[found], 2, group.orient_preservering);
            } else {
                *ts = subgroup_tspace[found];
                ts.counter = 1;
                ts.orient = group.orient_preservering;
            }
        }

        _unique_tspaces += unique_subgroups;
    }

    true
}

fn average_tspace(ts0: &TSpace, ts1: &TSpace, counter: usize, orient: bool) -> TSpace {
    let (mag_s, os, mag_t, ot);
    if ts0.mag_s == ts1.mag_s && ts0.mag_t == ts1.mag_t && ts0.os == ts1.os && ts0.ot == ts1.ot {
        mag_s = ts0.mag_s;
        mag_t = ts0.mag_t;
        os = ts0.os;
        ot = ts0.ot
    } else {
        mag_s = 0.5 * (ts0.mag_s + ts1.mag_s);
        mag_t = 0.5 * (ts0.mag_t + ts1.mag_t);
        os = (ts0.os + ts1.os).safe_normalize();
        ot = (ts0.ot + ts1.ot).safe_normalize();
    }
    TSpace {
        os,
        ot,
        mag_s,
        mag_t,
        counter,
        orient,
    }
}

fn eval_tspace<I: Geometry>(
    face_indices: &[usize],
    indices: &[usize],
    triangles: &[Triangle],
    geometry: &mut I,
    vertex_representitive: usize,
) -> TSpace {
    let mut angle_sum = 0.0;
    let mut os = Vec3::new(0.0, 0.0, 0.0);
    let mut mag_s = 0.0;
    let mut ot = Vec3::new(0.0, 0.0, 0.0);
    let mut mag_t = 0.0;

    for &face in face_indices {
        if triangles[face].flag & 4 == 0 {
            let idx = if indices[3 * face] == vertex_representitive {
                [2, 0, 1]
            } else if indices[3 * face + 1] == vertex_representitive {
                [0, 1, 2]
            } else if indices[3 * face + 2] == vertex_representitive {
                [1, 2, 0]
            } else {
                continue;
            };

            let i0 = indices[3 * face + idx[0]];
            let i1 = indices[3 * face + idx[1]];
            let i2 = indices[3 * face + idx[2]];

            let p0 = get_position(geometry, i0);
            let p1 = get_position(geometry, i1);
            let p2 = get_position(geometry, i2);

            let n = get_normal(geometry, i1);

            let xos = (triangles[face].os - n.dot(triangles[face].os) * n).safe_normalize();
            let xot = (triangles[face].ot - n.dot(triangles[face].ot) * n).safe_normalize();

            let v1 = p0 - p1;
            let v1 = (v1 - n.dot(v1) * n).safe_normalize();

            let v2 = p2 - p1;
            let v2 = (v2 - n.dot(v2) * n).safe_normalize();

            let cos = v1.dot(v2).clamp(-1.0, 1.0);

            let angle = cos.acos();

            os = xos + (angle * xos);
            ot = xot + (angle * xot);
            mag_s += angle * triangles[face].mag_s;
            mag_t += angle * triangles[face].mag_t;
            angle_sum += angle
        }
    }

    let os = os.safe_normalize();
    let ot = ot.safe_normalize();

    if angle_sum > 0.0 {
        mag_s /= angle_sum;
        mag_t /= angle_sum;
    }

    TSpace {
        mag_s,
        mag_t,
        os,
        ot,
        counter: 0,
        orient: false,
    }
}

fn build4_rule_groups(
    triangles: &mut [Triangle],
    groups: &mut [Group],
    group_triange_buffer: &mut [usize],
    indices: &[usize],
) -> usize {
    let mut active_groups = 0;
    let mut offset = 0;
    for face_index in 0..triangles.len() {
        if triangles[face_index].flag & 4 != 0 {
            continue;
        }

        for i in 0..3 {
            if triangles[face_index].assigned_group[i].is_null() {
                triangles[face_index].assigned_group[i] = &mut groups[active_groups] as *mut Group;

                let group = unsafe { &mut *triangles[face_index].assigned_group[i] };
                group.vertex_representitive = indices[face_index * 3 + i];
                group.orient_preservering = triangles[face_index].flag & 8 != 0;

                group.start = offset;
                group.end = offset;

                active_groups += 1;

                group_triange_buffer[group.end] = face_index;
                group.end += 1;

                let or_pre = triangles[face_index].flag & 8 != 0;
                let right = if i > 0 { i - 1 } else { 2 };
                let neigh_index_l = triangles[face_index].face_neighbors[i];
                let neigh_index_r = triangles[face_index].face_neighbors[right];
                if let Some(index) = neigh_index_l {
                    let _answer =
                        assign_recur(indices, triangles, index, group, group_triange_buffer);
                    let or_pre2 = triangles[index].flag & 8 != 0;
                    let _diff = or_pre != or_pre2;
                }
                if let Some(index) = neigh_index_r {
                    let _answer =
                        assign_recur(indices, triangles, index, group, group_triange_buffer);
                    let or_pre2 = triangles[index].flag & 8 != 0;
                    let _diff = or_pre != or_pre2;
                }
                offset += group.end - group.start;
            }
        }
    }

    active_groups
}

// ///////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////
fn assign_recur(
    indices: &[usize],
    triangles: &mut [Triangle],
    index: usize,
    group: &mut Group,
    group_triange_buffer: &mut [usize],
) -> bool {
    let mut info = &mut triangles[index];

    // track down vertex
    let vert_rep = group.vertex_representitive;
    let verts = &indices[3 * index..];
    let i = if verts[0] == vert_rep {
        0
    } else if verts[1] == vert_rep {
        1
    } else if verts[2] == vert_rep {
        2
    } else {
        return false;
    };

    if info.assigned_group[i] == group {
        return true;
    } else if !info.assigned_group[i].is_null() {
        return false;
    }

    if info.flag & 4 != 0
        && info.assigned_group[0].is_null()
        && info.assigned_group[1].is_null()
        && info.assigned_group[2].is_null()
    {
        info.flag &= !8;
        info.flag |= if group.orient_preservering { 8 } else { 0 };
    }

    let orient = info.flag & 8 != 0;
    if orient != group.orient_preservering {
        return false;
    }

    group_triange_buffer[group.end] = index;
    group.end += 1;

    info.assigned_group[i] = group;

    let neigh_index_l = info.face_neighbors[i];
    let neigh_index_r = info.face_neighbors[(if i > 0 { i - 1 } else { 2 })];
    if let Some(index) = neigh_index_l {
        assign_recur(indices, triangles, index, group, group_triange_buffer);
    }
    if let Some(index) = neigh_index_r {
        assign_recur(indices, triangles, index, group, group_triange_buffer);
    }

    true
}

fn init_tri_info<I: Geometry>(triangles: &mut [Triangle], indices: &[usize], geometry: &mut I) {
    for face in triangles.iter_mut() {
        face.os = Vec3::zero();
        face.ot = Vec3::zero();
        face.mag_s = 0.0;
        face.mag_t = 0.0;
        face.flag |= 4;
        for i in 0..3 {
            face.face_neighbors[i] = None;
            face.assigned_group[i] = null_mut();
        }
    }

    for face in 0..triangles.len() {
        let v1 = get_position(geometry, indices[face * 3]);
        let v2 = get_position(geometry, indices[face * 3 + 1]);
        let v3 = get_position(geometry, indices[face * 3 + 2]);

        let t1 = get_tex_coord(geometry, indices[face * 3]);
        let t2 = get_tex_coord(geometry, indices[face * 3 + 1]);
        let t3 = get_tex_coord(geometry, indices[face * 3 + 2]);

        let t21 = t2 - t1;
        let t31 = t3 - t1;

        let d1 = v2 - v1;
        let d2 = v3 - v1;

        let os = (t31.y * d1) - (t21.y * d2);
        let ot = (-t31.x * d1) + (t21.x * d2);

        let info = &mut triangles[face];
        let signed_area_st_x2 = t21.x * t31.y - t21.y * t31.x;
        info.flag |= if signed_area_st_x2 > 0.0 { 8 } else { 0 };

        if Vec3::not_zero(signed_area_st_x2) {
            let abs_area = signed_area_st_x2.abs();

            let len_os = os.length();
            let len_ot = ot.length();

            let s = if info.flag & 8 == 0 { -1.0 } else { 1.0 };

            if Vec3::not_zero(len_os) {
                info.os = (s / len_os) * os
            }
            if Vec3::not_zero(len_ot) {
                info.ot = (s / len_ot) * ot
            }
            info.mag_s = len_os / abs_area;
            info.mag_t = len_ot / abs_area;
            if Vec3::not_zero(info.mag_s) && Vec3::not_zero(info.mag_t) {
                info.flag &= !4
            }
        }
    }

    let mut t = 0;
    while t < triangles.len() - 1 {
        let fo_a = triangles[t].original_face;
        let fo_b = triangles[t + 1].original_face;
        if fo_a == fo_b {
            let is_deg_a = triangles[t].flag & 1 != 0;
            let is_deg_b = triangles[t + 1].flag & 1 != 0;
            if !(is_deg_a || is_deg_b) {
                let orient_a = triangles[t].flag & 8 != 0;
                let orient_b = triangles[t + 1].flag & 8 != 0;
                if orient_a != orient_b {
                    let choose_orient_first_tri = triangles[t + 1].flag & 4 != 0
                        || calc_tex_area(geometry, &indices[t * 3..])
                            >= calc_tex_area(geometry, &indices[(t + 1) * 3..]);

                    let t0 = if choose_orient_first_tri { t } else { t + 1 };
                    let t1 = if choose_orient_first_tri { t + 1 } else { t };
                    triangles[t1].flag &= !8;
                    triangles[t1].flag |= triangles[t0].flag & 8;
                }
            }
            t += 2
        } else {
            t += 1
        }
    }

    let mut edges = vec![Edge::zero(); triangles.len() * 3];
    build_neighbors_fast(triangles, &mut edges, indices);
}

fn build_neighbors_fast(triangles: &mut [Triangle], edges: &mut [Edge], indices: &[usize]) {
    // build array of edges
    // could replace with a random seed?
    let seed = 39871946;

    for f in 0..triangles.len() {
        for i in 0..3 {
            let i0 = indices[f * 3 + i];
            let i1 = indices[(f * 3 + if i < 2 { i + 1 } else { 0 })];
            edges[f * 3 + i] = Edge {
                i0: if i0 < i1 { i0 } else { i1 },
                i1: if i0 >= i1 { i0 } else { i1 },
                f,
            };
        }
    }

    quick_sort_edges(edges, 0, triangles.len() * 3 - 1, 0, seed);

    let entries = 3 * triangles.len();

    let mut cur_start_index = 0;
    for i in 1..entries {
        if edges[cur_start_index].i0 != edges[i].i0 {
            let l = cur_start_index;
            let r = i - 1;
            cur_start_index = i;
            quick_sort_edges(edges, l, r, 1, seed);
        }
    }

    let mut cur_start_index = 0;
    for i in 1..entries {
        if edges[cur_start_index].i0 != edges[i].i0 || edges[cur_start_index].i1 != edges[i].i1 {
            let l = cur_start_index;
            let r = i - 1;
            cur_start_index = i;
            quick_sort_edges(edges, l, r, 2, seed);
        }
    }

    for i in 0..entries {
        let Edge { i0, i1, f } = edges[i];

        let (edgenum_a, i0_a, i1_a) = get_edge(&indices[f * 3..], i0, i1);

        let unassigned_a = triangles[f].face_neighbors[edgenum_a].is_none();
        if unassigned_a {
            let mut j = i + 1;
            let mut edgenum_b = None;
            while j < entries && i0 == edges[j].i0 && i1 == edges[j].i1 && edgenum_b.is_none() {
                let t = edges[j].f;
                let (edgenum, i1_b, i0_b) = get_edge(&indices[t * 3..], edges[j].i0, edges[j].i1);
                let unassigned_b = triangles[t].face_neighbors[edgenum].is_none();
                if i0_a == i0_b && i1_a == i1_b && unassigned_b {
                    edgenum_b = Some(edgenum)
                } else {
                    j += 1
                }
            }
            if let Some(edgenum_b) = edgenum_b {
                let t = edges[j].f;
                triangles[f].face_neighbors[edgenum_a] = Some(t);
                triangles[t].face_neighbors[edgenum_b] = Some(f);
            }
        }
    }
}

fn get_edge(indices: &[usize], i0: usize, i1: usize) -> (usize, usize, usize) {
    if indices[0] == i0 || indices[0] == i1 {
        if indices[1] == i0 || indices[1] == i1 {
            (0, indices[0], indices[1])
        } else {
            (2, indices[2], indices[0])
        }
    } else {
        (1, indices[1], indices[2])
    }
}

// ///////////////////////////////////////////////////////////////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////////////////////
fn quick_sort_edges(edges: &mut [Edge], left: usize, right: usize, channel: usize, seed: u32) {
    // early out
    let elements = right - left + 1;
    match elements.cmp(&2) {
        Ordering::Less => return,
        Ordering::Equal => {
            if edges[left].array(channel) > edges[right].array(channel) {
                edges.swap(left, right)
            }
            return;
        }
        Ordering::Greater => (),
    }

    // Random
    let t = seed & 31;
    let t = seed.rotate_left(t) | seed.rotate_right((32u32).wrapping_sub(t));
    let seed = seed.wrapping_add(t).wrapping_add(3);

    // Random end

    let mut l = left;
    let mut r = right;
    let index = seed.wrapping_rem(elements as u32) as usize;
    let mid = edges[l + index].array(channel);
    loop {
        while edges[l].array(channel) < mid {
            l += 1
        }
        while edges[r].array(channel) > mid {
            r -= 1
        }
        if l <= r {
            edges.swap(l, r);
            l += 1;
            r -= 1
        }
        if l > r {
            break;
        }
    }

    if left < r {
        quick_sort_edges(edges, left, r, channel, seed);
    }
    if l < right {
        quick_sort_edges(edges, l, right, channel, seed);
    }
}

// returns the texture area times 2
fn calc_tex_area<I: Geometry>(geometry: &mut I, indices: &[usize]) -> f32 {
    let t1 = get_tex_coord(geometry, indices[0]);
    let t2 = get_tex_coord(geometry, indices[1]);
    let t3 = get_tex_coord(geometry, indices[2]);

    let t21 = t2 - t1;
    let t31 = t3 - t1;

    let signed_area_st_x2 = t21.x * t31.y - t21.y * t31.x;
    if signed_area_st_x2 < 0.0 {
        -signed_area_st_x2
    } else {
        signed_area_st_x2
    }
}

// degen triangles
fn degen_prologue(triangles: &mut [Triangle], indices: &mut [usize], num_triangles: usize) {
    // locate quads with only one good triangle
    let mut t = 0;
    while t < triangles.len() - 1 {
        let fo_a = triangles[t].original_face;
        let fo_b = triangles[t + 1].original_face;
        if fo_a == fo_b {
            let is_deg_a = triangles[t].flag & 1 != 0;
            let is_deg_b = triangles[t + 1].flag & 1 != 0;
            if is_deg_a ^ is_deg_b {
                triangles[t].flag |= 2;
                triangles[t + 1].flag |= 2;
            }
            t += 2
        } else {
            t += 1
        }
    }

    let mut next_good_triangle_search_index = 1;
    let mut t = 0;
    let mut still_finding_good_ones = true;
    while t < num_triangles && still_finding_good_ones {
        let is_good = triangles[t].flag & 1 == 0;
        if is_good {
            if next_good_triangle_search_index < t + 2 {
                next_good_triangle_search_index = t + 2
            }
        } else {
            let mut just_a_degenerate = true;
            while just_a_degenerate && next_good_triangle_search_index < triangles.len() {
                let is_good = triangles[next_good_triangle_search_index].flag & 1 == 0;
                if is_good {
                    just_a_degenerate = false
                } else {
                    next_good_triangle_search_index += 1
                }
            }

            let (t0, t1) = (t, next_good_triangle_search_index);
            next_good_triangle_search_index += 1;
            if !just_a_degenerate {
                for i in 0..3 {
                    indices.swap(t0 * 3 + i, t1 * 3 + i);
                }
                triangles.swap(t0, t1);
            } else {
                still_finding_good_ones = false
            }
        }

        if still_finding_good_ones {
            t += 1
        }
    }
}

fn generate_shared_vertices_index_list<I: Geometry>(indices: &mut [usize], geometry: &mut I) {
    let mut min = get_position(geometry, 0);
    let mut max = min;

    for &index in &indices[1..] {
        let pt = get_position(geometry, index);
        if min.x > pt.x {
            min.x = pt.x
        } else if max.x < pt.x {
            max.x = pt.x
        }
        if min.y > pt.y {
            min.y = pt.y
        } else if max.y < pt.y {
            max.y = pt.y
        }
        if min.z > pt.z {
            min.z = pt.z
        } else if max.z < pt.z {
            max.z = pt.z
        }
    }

    let dim = max - min;

    let (channel, min, max) = if dim.y > dim.x && dim.y > dim.z {
        (1, min.y, max.y)
    } else if dim.z > dim.x {
        (2, min.z, max.z)
    } else {
        (0, min.x, max.x)
    };

    let mut hash_table = vec![0usize; indices.len()];
    let mut hash_offsets = vec![0usize; CELLS];
    let mut hash_count = vec![0usize; CELLS];
    let mut hash_count2 = vec![0usize; CELLS];

    for &index in indices.iter() {
        let pt = get_position(geometry, index);
        let value = match channel {
            0 => pt.x,
            1 => pt.y,
            _ => pt.z,
        };
        let cell = find_grid_cell(min, max, value);
        hash_count[cell] += 1;
    }

    hash_offsets[0] = 0;
    for k in 1..CELLS {
        hash_offsets[k] = hash_offsets[k - 1] + hash_count[k - 1];
    }

    for (i, &index) in indices.iter().enumerate() {
        let pt = get_position(geometry, index);
        let value = match channel {
            0 => pt.x,
            1 => pt.y,
            _ => pt.z,
        };
        let cell = find_grid_cell(min, max, value);
        hash_table[hash_offsets[cell] + hash_count2[cell]] = i;
        hash_count2[cell] += 1;
    }

    let max_count = hash_count.iter().copied().max().unwrap_or(0);

    let mut tmp = vec![TmpVert::zero(); max_count];
    for k in 0..CELLS {
        // extract table of cell k and amount of entries in it
        let table = &hash_table[hash_offsets[k]..];
        let entries = hash_count[k];
        if entries >= 2 {
            for e in 0..entries {
                let index = table[e];
                tmp[e] = TmpVert {
                    vert: get_position(geometry, indices[index]).into(),
                    index,
                };
            }
            merge_verts_fast(indices, &mut tmp, geometry, 0, entries - 1);
        }
    }
}

fn merge_verts_fast<I: Geometry>(
    triangles: &mut [usize],
    tmp: &mut [TmpVert],
    geometry: &mut I,
    in_l: usize,
    in_r: usize,
) {
    // make bbox
    let mut min = [0.0; 3];
    min.clone_from_slice(&tmp[in_l].vert);

    let mut max = min;
    for tmp in &tmp[in_l + 1..=in_r] {
        for c in 0..3 {
            let value = tmp.vert[c];
            if min[c] > value {
                min[c] = value;
            } else if max[c] < value {
                max[c] = value;
            }
        }
    }

    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];

    let channel = if dy > dx && dy > dz {
        1
    } else if dz > dx {
        2
    } else {
        0
    };

    let step = 0.5 * (max[channel] + min[channel]);
    if step >= max[channel] || step <= min[channel] {
        for l in in_l..=in_r {
            let i = tmp[l].index;

            let index = triangles[i];
            let p1 = get_position(geometry, index);
            let n1 = get_normal(geometry, index);
            let t1 = get_tex_coord(geometry, index);

            let mut l2 = in_l;
            let mut i2_rec = None;
            while l2 < l && i2_rec.is_none() {
                let i2 = tmp[l2].index;

                let index = triangles[i2];
                let p2 = get_position(geometry, index);
                let n2 = get_normal(geometry, index);
                let t2 = get_tex_coord(geometry, index);

                if p1 == p2 && n1 == n2 && t1 == t2 {
                    i2_rec = Some(i2);
                } else {
                    l2 += 1
                }
            }
            if let Some(i2) = i2_rec {
                triangles[i] = triangles[i2]
            }
        }
    } else {
        let mut il = in_l;
        let mut ir = in_r;
        while il < ir {
            let mut ready_left_swap = false;
            while !ready_left_swap && il < ir {
                ready_left_swap = tmp[il].vert[channel] >= step;
                if !ready_left_swap {
                    il += 1
                }
            }
            let mut ready_right_swap = false;
            while !ready_right_swap && il < ir {
                ready_right_swap = tmp[ir].vert[channel] < step;
                if !ready_right_swap {
                    ir -= 1
                }
            }
            if ready_left_swap && ready_right_swap {
                tmp.swap(il, ir);
                il += 1;
                ir -= 1
            }
        }
        if il == ir {
            let ready_right_swap = tmp[ir].vert[channel] < step;
            if ready_right_swap {
                il += 1
            } else {
                ir -= 1
            }
        }
        if in_l < ir {
            merge_verts_fast(triangles, tmp, geometry, in_l, ir);
        }
        if il < in_r {
            merge_verts_fast(triangles, tmp, geometry, il, in_r);
        }
    };
}

const CELLS: usize = 2048;

// it is IMPORTANT that this function is called to evaluate the hash since
// inlining could potentially reorder instructions and generate different
// results for the same effective input value fVal.
#[inline(never)]
fn find_grid_cell(min: f32, max: f32, value: f32) -> usize {
    (CELLS as f32 * ((value - min) / (max - min))).clamp(0.0, CELLS as f32 - 1.0) as usize
}

fn generate_initial_vertices_index_list<I: Geometry>(
    triangles: &mut [Triangle],
    indices: &mut [usize],
    geometry: &mut I,
) -> usize {
    let mut tspaces_offset = 0;
    let mut dst = 0;
    for face in 0..geometry.num_faces() {
        let verts = geometry.num_vertices_of_face(face);
        if verts != 3 && verts != 4 {
            continue;
        }

        triangles[dst].original_face = face;
        triangles[dst].tspaces_offset = tspaces_offset;

        if verts == 3 {
            triangles[dst].vert_num = [0, 1, 2];
            indices[dst * 3] = face_vert_to_index(face, 0);
            indices[dst * 3 + 1] = face_vert_to_index(face, 1);
            indices[dst * 3 + 2] = face_vert_to_index(face, 2);
            dst += 1
        } else {
            triangles[dst + 1].original_face = face;
            triangles[dst + 1].tspaces_offset = tspaces_offset;

            let i0 = face_vert_to_index(face, 0);
            let i1 = face_vert_to_index(face, 1);
            let i2 = face_vert_to_index(face, 2);
            let i3 = face_vert_to_index(face, 3);

            let t0 = get_tex_coord(geometry, i0);
            let t1 = get_tex_coord(geometry, i1);
            let t2 = get_tex_coord(geometry, i2);
            let t3 = get_tex_coord(geometry, i3);

            let dist_sq_02 = (t2 - t0).length_squared();
            let dist_sq_13 = (t3 - t1).length_squared();

            let is_02 = if dist_sq_02 < dist_sq_13 {
                true
            } else if dist_sq_13 < dist_sq_02 {
                false
            } else {
                let p0 = get_position(geometry, i0);
                let p1 = get_position(geometry, i1);
                let p2 = get_position(geometry, i2);
                let p3 = get_position(geometry, i3);
                (p3 - p1).length_squared() >= (p2 - p0).length_squared()
            };

            if is_02 {
                triangles[dst].vert_num = [0, 1, 2];
                indices[dst * 3] = i0;
                indices[dst * 3 + 1] = i1;
                indices[dst * 3 + 2] = i2;
                dst += 1;

                triangles[dst].vert_num = [0, 2, 3];
                indices[dst * 3] = i0;
                indices[dst * 3 + 1] = i2;
                indices[dst * 3 + 2] = i3;
                dst += 1
            } else {
                triangles[dst].vert_num = [0, 1, 3];
                indices[dst * 3] = i0;
                indices[dst * 3 + 1] = i1;
                indices[dst * 3 + 2] = i3;
                dst += 1;

                triangles[dst].vert_num = [1, 2, 3];
                indices[dst * 3] = i1;
                indices[dst * 3 + 1] = i2;
                indices[dst * 3 + 2] = i3;
                dst += 1
            }
        }

        tspaces_offset += verts
    }

    for info in triangles.iter_mut() {
        info.flag = 0;
    }

    tspaces_offset
}
