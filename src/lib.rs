mod generated;
mod vector;

/// The interface by which mikktspace interacts with your geometry.
pub trait Geometry {
    /// Returns the number of faces.
    fn num_faces(&self) -> usize;

    /// Returns the number of vertices of a face.
    fn num_vertices_of_face(&self, face: usize) -> usize;

    /// Returns the position of a vertex.
    fn position(&self, face: usize, vert: usize) -> [f32; 3];

    /// Returns the normal of a vertex.
    fn normal(&self, face: usize, vert: usize) -> [f32; 3];

    /// Returns the texture coordinate of a vertex.
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2];

    /// Sets the generated tangent for a vertex.
    /// Leave this function unimplemented if you are implementing
    /// `set_tangent_encoded`.
    fn set_tangent(
        &mut self,
        tangent: [f32; 3],
        _bitangent: [f32; 3],
        _mag_st: [f32; 2],
        bitangent_preserves_orientation: bool,
        face: usize,
        vert: usize,
    ) {
        let sign = if bitangent_preserves_orientation {
            1.0
        } else {
            -1.0
        };
        self.set_tangent_encoded([tangent[0], tangent[1], tangent[2], sign], face, vert);
    }

    /// Sets the generated tangent for a vertex with its bi-tangent encoded as the 'W' (4th)
    /// component in the tangent. The 'W' component marks if the bi-tangent is flipped. This
    /// is called by the default implementation of `set_tangent`; therefore, this function will
    /// not be called by the crate unless `set_tangent` is unimplemented.
    fn set_tangent_encoded(&mut self, _tangent: [f32; 4], _face: usize, _vert: usize) {}
}

/// Generates tangents for the input geometry.
///
/// # Errors
///
/// Returns `false` if the geometry is unsuitable for tangent generation including,
/// but not limited to, lack of vertices.
pub fn generate_tangents<I: Geometry>(geometry: &mut I) -> bool {
    generated::generate_tangent_space(geometry, 180.0)
}

fn get_position<I: Geometry>(geometry: &mut I, index: usize) -> crate::vector::Vec3 {
    let (face, vert) = index_to_face_vert(index);
    geometry.position(face, vert).into()
}

fn get_tex_coord<I: Geometry>(geometry: &mut I, index: usize) -> crate::vector::Vec3 {
    let (face, vert) = index_to_face_vert(index);
    let [u, v] = geometry.tex_coord(face, vert);
    crate::vector::Vec3::new(u, v, 1.0)
}

fn get_normal<I: Geometry>(geometry: &mut I, index: usize) -> crate::vector::Vec3 {
    let (face, vert) = index_to_face_vert(index);
    geometry.normal(face, vert).into()
}

fn index_to_face_vert(index: usize) -> (usize, usize) {
    (index >> 2, index & 0x3)
}

fn face_vert_to_index(face: usize, vert: usize) -> usize {
    face << 2 | vert & 0x3
}
