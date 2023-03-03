#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geometrycentral/surface/halfedge_factories.h"
#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_centers.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include <chrono>
#include <sstream>

using namespace geometrycentral;
using namespace geometrycentral::surface;
namespace py = pybind11;

// Geometry-central data
std::unique_ptr<HalfedgeMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

// Algorithm parameters for Vector Heat method
float tCoef = 1.0;
std::unique_ptr<VectorHeatMethodSolver> solver;

// HELPER FUNCTIONS ------------------------------------------------------------

// Loads a mesh from a NumPy array
std::tuple<std::unique_ptr<HalfedgeMesh>,
           std::unique_ptr<VertexPositionGeometry>>
loadMesh_np(Eigen::MatrixXd &pos,
            Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> &faces) {

  // Set vertex positions
  std::vector<Vector3> vertexPositions(pos.rows());
  for (size_t i = 0; i < pos.rows(); i++) {
    vertexPositions[i][0] = pos(i, 0);
    vertexPositions[i][1] = pos(i, 1);
    vertexPositions[i][2] = pos(i, 2);
  }

  // Get face list
  std::vector<std::vector<size_t>> faceIndices(faces.rows());
  for (size_t i = 0; i < faces.rows(); i++) {
    faceIndices[i] = {faces(i, 0), faces(i, 1), faces(i, 2)};
  }

  return makeHalfedgeAndGeometry(faceIndices, vertexPositions);
}

void prepare(Eigen::MatrixXd &pos,
             Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> &faces) {
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);
}

// Precompute parallel transport and logarithmic map for a given neighborhood.
Eigen::MatrixXd get_laplacian() {
  return Eigen::MatrixXd(geometry->cotanLaplacian);
}

// Returns an extrinsic representation of the tangent frame being used
// internally, as X/Y/N vectors.
DenseMatrix<double> get_normals() {
  geometry->requireVertexNormals();
  Eigen::MatrixXd normals(mesh->nVertices(), 3);
  size_t i = 0;
  for (Vertex v : mesh->vertices()) {
    geometrycentral::Vector3 normal = geometry->vertexNormals[v];
    normals(i, 0) = normal.x;
    normals(i, 1) = normal.y;
    normals(i, 2) = normal.z;
    i++;
  }

  return normals;
}

// Returns an extrinsic representation of the tangent frame being used
// internally, as X/Y/N vectors.
std::tuple<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>
get_tangent_frames() {

  // Just in case we don't already have it
  geometry->requireVertexNormals();
  geometry->requireVertexTangentBasis();

  Eigen::MatrixXd basisX(mesh->nVertices(), 3);
  Eigen::MatrixXd basisY(mesh->nVertices(), 3);
  Eigen::MatrixXd normals(mesh->nVertices(), 3);
  size_t i = 0;
  for (Vertex v : mesh->vertices()) {
    geometrycentral::Vector3 x = geometry->vertexTangentBasis[v][0];
    geometrycentral::Vector3 y = geometry->vertexTangentBasis[v][1];
    geometrycentral::Vector3 normal = geometry->vertexNormals[v];
    basisX(i, 0) = x.x;
    basisX(i, 1) = x.y;
    basisX(i, 2) = x.z;
    basisY(i, 0) = y.x;
    basisY(i, 1) = y.y;
    basisY(i, 2) = y.z;
    normals(i, 0) = normal.x;
    normals(i, 1) = normal.y;
    normals(i, 2) = normal.z;
    i++;
  }

  return std::tuple<DenseMatrix<double>, DenseMatrix<double>,
                    DenseMatrix<double>>(basisX, basisY, normals);
}
// Precompute parallel transport and logarithmic map for a given neighborhood.
Eigen::MatrixXd
precomputeHarmonic(Vertex &sourceV,
                   Eigen::Matrix<size_t, Eigen::Dynamic, 1> targetVs,
                   Eigen::Matrix<size_t, Eigen::Dynamic, 1> &sample_points) {
  if (solver == nullptr) {
    solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));
  }

  // Coordinate systems are aligned to smoothed principal curvature directions
  // Vector2 sourcePrincipalCurvature =
  //     geometry->vertexPrincipalCurvatureDirections[sourceV].normalize();

  // To compute parallel transport from point i (targetV) to point j (sourceV),
  // we transport the x-axis (the principal curvature direction) from sourceV to
  // targetV.

  // First, set the source vectors to the principal curvature directions.
  std::vector<std::tuple<SurfacePoint, Vector2>> points;
  points.emplace_back(sourceV, Vector2{1.0, 0.0});

  // Then, compute parallel transport of source vectors.
  VertexData<Vector2> connection = solver->transportTangentVectors(points);
  // And compute the logarithmic map from point i to j
  VertexData<Vector2> logmap = solver->computeLogMap(sourceV);

  // Store the results in an Eigen matrix, which can be accessed as a NumPy
  // array.
  Eigen::MatrixXd res(targetVs.rows(), 4);

  // For every target point
  for (size_t i = 0; i < targetVs.rows(); i++) {
    size_t v = sample_points(targetVs(i));

    // The original logarithmic map is computed with a coordinate system aligned
    // to the first edge. To align the logarithmic map to the principal
    // curvature directions, we rotate the logmap by the principal curvature
    // direction at the source point.
    // Vector2 targetCoords = logmap[v] / sourcePrincipalCurvature;

    // Likewise for parallel transport, but we rotate by the principal curvature
    // direction at the target point.
    // Vector2 targetPrincipalCurvature =
    //     geometry->vertexPrincipalCurvatureDirections[v].normalize();
    // Vector2 targetConnection = connection[v] / targetPrincipalCurvature;

    // Store the parallel transport (connection) and logarithmic map.
    // res(i, 0) = targetConnection.x;
    // res(i, 1) = targetConnection.y;
    // res(i, 2) = targetCoords.x;
    // res(i, 3) = targetCoords.y;
    res(i, 0) = connection[v].x;
    res(i, 1) = connection[v].y;
    res(i, 2) = logmap[v].x;
    res(i, 3) = logmap[v].y;
  }

  return res;
}

// Precompute parallel transport and logarithmic map for a given neighborhood
// using surface points.
Eigen::MatrixXd precomputeHarmonicSurface(
    SurfacePoint &sourcePoint,
    Eigen::Matrix<size_t, Eigen::Dynamic, 1> target_points,
    Eigen::Matrix<size_t, Eigen::Dynamic, 1> &face_idx,
    Eigen::MatrixXd &face_coords,
    Eigen::Matrix<size_t, Eigen::Dynamic, 1> &sample_points) {

  // Setup solver for Vector Heat Method.
  if (solver == nullptr) {
    solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));
  }

  // First, set the source vectors to the principal curvature directions.
  std::vector<std::tuple<SurfacePoint, Vector2>> points;
  points.emplace_back(sourcePoint, Vector2{1.0, 0.0});

  // Then, compute parallel transport of source vectors.
  VertexData<Vector2> connection = solver->transportTangentVectors(points);
  // And compute the logarithmic map from point i to j
  VertexData<Vector2> logmap = solver->computeLogMap(sourcePoint);

  // Store into a matrix compatible with python
  Eigen::MatrixXd res(target_points.rows(), 4);
  for (size_t row = 0; row < target_points.rows(); row++) {
    auto i = sample_points(target_points(row));
    auto face = mesh->face(face_idx(i));
    auto coords = face_coords.row(i);
    auto targetV = SurfacePoint(face, {coords.x(), coords.y(), coords.z()});

    auto target_connection = targetV.interpolate(connection);
    auto target_logmap = targetV.interpolate(logmap);

    res(row, 0) = target_connection.x;
    res(row, 1) = target_connection.y;
    res(row, 2) = target_logmap.x;
    res(row, 3) = target_logmap.y;
  }
  return res;
}

// Precompute parallel transport and logarithmic map for a given neighborhood.
Eigen::MatrixXd logmapVertex(size_t source_point) {
  geometry->requireVertexIndices();
  geometry->requireVertexLumpedMassMatrix();
  geometry->requireVertexPrincipalCurvatureDirections();

  Vertex sourceV = mesh->vertex(source_point);

  // Setup solver for Vector Heat Method.
  solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));

  // First, set the source vectors to the principal curvature directions.
  std::vector<std::tuple<SurfacePoint, Vector2>> points;
  points.emplace_back(sourceV, Vector2{1.0, 0.0});

  // And compute the logarithmic map from point i to j
  VertexData<Vector2> logmap = solver->computeLogMap(sourceV);

  // Store into a matrix compatible with python
  Eigen::MatrixXd res(mesh->nVertices(), 2);
  for (size_t i = 0; i < mesh->nVertices(); i++) {
    res(i, 0) = logmap[i].x;
    res(i, 1) = logmap[i].y;
  }

  return res;
}

Eigen::MatrixXd logmapSurfacePoint(size_t source_face,
                                   Eigen::Vector3d face_coords) {
  geometry->requireVertexIndices();
  geometry->requireVertexLumpedMassMatrix();
  geometry->requireVertexPrincipalCurvatureDirections();
  Face face = mesh->face(source_face);
  auto sourceV =
      SurfacePoint(face, {face_coords.x(), face_coords.y(), face_coords.z()});

  // Setup solver for Vector Heat Method.
  solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));

  // First, set the source vectors to the principal curvature directions.
  std::vector<std::tuple<SurfacePoint, Vector2>> points;
  points.emplace_back(sourceV, Vector2{1.0, 0.0});

  // And compute the logarithmic map from point i to j
  VertexData<Vector2> logmap = solver->computeLogMap(sourceV);

  // Store into a matrix compatible with python
  Eigen::MatrixXd res(mesh->nVertices(), 2);
  for (size_t i = 0; i < mesh->nVertices(); i++) {
    res(i, 0) = logmap[i].x;
    res(i, 1) = logmap[i].y;
  }

  return res;
}

// PRECOMPUTATION for HSN
// ------------------------------------------------------------

// Precomputes the logarithmic map and parallel transport, given a mesh.
// The mesh should be provides as a NumPy array of vertex positions and a NumPy
// array of face indices. Additionally, one should provide a NumPy array of edge
// indices (source, target), a NumPy array with the degree of every source
// vertex, and indices of sampled points to return values for.
Eigen::MatrixXd
precompute(Eigen::MatrixXd &pos,
           Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> &faces,
           Eigen::Matrix<size_t, Eigen::Dynamic, 2> &edge_index,
           Eigen::Matrix<size_t, Eigen::Dynamic, 1> degree,
           Eigen::Matrix<size_t, Eigen::Dynamic, 1> &sample_points) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  geometry->requireVertexIndices();
  geometry->requireVertexLumpedMassMatrix();
  geometry->requireVertexPrincipalCurvatureDirections();

  // Setup solver for Vector Heat Method.
  solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));
  // Store the results in an Eigen matrix, which can be accessed as a NumPy
  // array.
  Eigen::MatrixXd res(edge_index.rows(), 4);
  size_t index = 0;
  // For each sampled point:
  for (size_t row = 0; row < sample_points.rows(); row++) {
    Vertex v = mesh->vertex(sample_points(row));

    // Compute parallel transport and logarithmic map for neighborhood.
    res.block(index, 0, degree(row), 4) = precomputeHarmonic(
        v, edge_index.block(index, 1, degree(row), 1), sample_points);
    index += degree(row);
  }

  geometry->unrequireVertexPrincipalCurvatureDirections();
  geometry->unrequireVertexLumpedMassMatrix();
  geometry->unrequireVertexIndices();
  return res;
}

// PRECOMPUTATION for HSN
// ------------------------------------------------------------

// Precomputes the logarithmic map and parallel transport, given a mesh.
// The mesh should be provides as a NumPy array of vertex positions and a NumPy
// array of face indices. Additionally, one should provide a NumPy array of edge
// indices (source, target), a NumPy array with the degree of every source
// vertex, and indices of sampled points to return values for.
Eigen::MatrixXd
precomputeSurface(Eigen::MatrixXd &pos,
                  Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> &faces,
                  Eigen::Matrix<size_t, Eigen::Dynamic, 1> &face_idx,
                  Eigen::MatrixXd &face_coords,
                  Eigen::Matrix<size_t, Eigen::Dynamic, 2> &edge_index,
                  Eigen::Matrix<size_t, Eigen::Dynamic, 1> degree,
                  Eigen::Matrix<size_t, Eigen::Dynamic, 1> &sample_points) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  geometry->requireVertexIndices();
  geometry->requireVertexLumpedMassMatrix();
  geometry->requireVertexPrincipalCurvatureDirections();

  // Setup solver for Vector Heat Method.
  solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));
  // Store the results in an Eigen matrix, which can be accessed as a NumPy
  // array.
  Eigen::MatrixXd res(edge_index.rows(), 4);
  size_t index = 0;
  // For each sampled point:
  for (size_t row = 0; row < sample_points.rows(); row++) {
    auto i = sample_points(row);
    auto face = mesh->face(face_idx(i));
    auto coords = face_coords.row(i);
    auto sourceV = SurfacePoint(face, {coords.x(), coords.y(), coords.z()});
    auto targets = edge_index.block(index, 1, degree(row), 1);
    // Compute parallel transport and logarithmic map for neighborhood.
    res.block(index, 0, degree(row), 4) = precomputeHarmonicSurface(
        sourceV, targets, face_idx, face_coords, sample_points);
    index += degree(row);
  }

  geometry->unrequireVertexPrincipalCurvatureDirections();
  geometry->unrequireVertexLumpedMassMatrix();
  geometry->unrequireVertexIndices();
  return res;
}

// Computes the vertex lumped mass matrix for each sampled vertex,
// automatically adding the weights from nearest geodesic neighbors.
Eigen::MatrixXd
weights(Eigen::MatrixXd &pos,
        Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> &faces,
        Eigen::Matrix<size_t, Eigen::Dynamic, 1> &sample_points,
        Eigen::Matrix<size_t, Eigen::Dynamic, 1> &labels) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  geometry->requireVertexIndices();
  geometry->requireVertexLumpedMassMatrix();

  // We use short-time heat diffusion to retrieve geodesic nearest neighbors.
  VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);

  // Set up indices of sampled points to diffuse.
  std::vector<std::tuple<SurfacePoint, double>> points;
  for (size_t row = 0; row < sample_points.rows(); row++) {
    points.emplace_back(SurfacePoint(mesh->vertex(sample_points(row))),
                        labels(row));
  }

  // Solve heat diffusion.
  VertexData<double> scalarExtension = vhmSolver.extendScalar(points);

  // Store the results in an Eigen matrix, which can be accessed as a NumPy
  // array.
  Eigen::MatrixXd res = Eigen::MatrixXd::Zero(sample_points.rows(), 1);
  // For each vertex:
  for (size_t row = 0; row < pos.rows(); row++) {
    size_t to_idx = std::lround(scalarExtension[mesh->vertex(row)]);

    // Clamp nearest neighbor index from heat diffusion to range [0, n_vertices]
    if (to_idx >= sample_points.rows()) {
      to_idx = sample_points.rows() - 1;
    } else if (to_idx < 0) {
      to_idx = 0;
    }

    // Add vertex lumped mass to nearest sampled vertex.
    res(to_idx) += geometry->vertexLumpedMassMatrix.coeff(row, row);
  }

  geometry->unrequireVertexLumpedMassMatrix();
  geometry->unrequireVertexIndices();

  return res;
}

// UTILITIES for HSN
// ------------------------------------------------------------

// Compute the surface area of a mesh.
double
surface_area(Eigen::MatrixXd &pos,
             Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> &faces) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  float surfaceArea = 0.0f;
  for (Face f : mesh->faces()) {
    surfaceArea += geometry->faceArea(f);
  }

  return surfaceArea;
}

// Compute geodesic nearest neighbors
Eigen::MatrixXd
nearest(Eigen::MatrixXd &pos,
        Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic> &faces,
        Eigen::Matrix<size_t, Eigen::Dynamic, 1> &selected_points,
        Eigen::Matrix<size_t, Eigen::Dynamic, 1> &labels) {

  // Load mesh
  std::tie(mesh, geometry) = loadMesh_np(pos, faces);

  geometry->requireVertexIndices();

  // We use short-time heat diffusion to retrieve geodesic nearest neighbors.
  VectorHeatMethodSolver vhmSolver(*geometry, 0.0001);

  // Set up indices of sampled points to diffuse.
  std::vector<std::tuple<SurfacePoint, double>> points;
  for (size_t row = 0; row < selected_points.rows(); row++) {
    points.emplace_back(SurfacePoint(mesh->vertex(selected_points(row))),
                        labels(row));
  }

  // Solve heat diffusion
  VertexData<double> scalarExtension = vhmSolver.extendScalar(points);

  // Store the results in an Eigen matrix, which can be accessed as a NumPy
  // array.
  Eigen::MatrixXd res(pos.rows(), 1);
  for (size_t row = 0; row < pos.rows(); row++) {
    res(row) = scalarExtension[mesh->vertex(row)];
  }

  geometry->unrequireVertexIndices();

  return res;
}

PYBIND11_MODULE(vectorheat, m) {
  m.doc() = R"pbdoc(
        Harmonic Surface Networks precomputation module.
        -----------------------

        .. currentmodule:: precomputation

        .. autosummary::
           :toctree: _generate

           add
           precompute
           diameter
    )pbdoc";

  m.def("get_laplacian", &get_laplacian, py::return_value_policy::copy,
        R"pbdoc(
        Get heatmethod laplacian
    )pbdoc");

  m.def("loadMesh", &prepare, py::return_value_policy::copy,
        R"pbdoc(
        Loads mesh
    )pbdoc");

  m.def("get_normals", &get_normals, py::return_value_policy::copy,
        R"pbdoc(
        Get normals
    )pbdoc");

  m.def("get_tangent_frames", &get_tangent_frames,
        py::return_value_policy::copy,
        R"pbdoc(
        Get tangent frames
    )pbdoc");

  m.def("precompute", &precompute, py::return_value_policy::copy, R"pbdoc(
        Precompute parallel transport and logarithmic map for meshes given by pos, face, edges and degree.
    )pbdoc");

  m.def("precomputeSurface", &precomputeSurface, py::return_value_policy::copy,
        R"pbdoc(
        Precompute parallel transport and logarithmic map for points on the surface of a mesh.
    )pbdoc");

  m.def("surfaceMap", &logmapSurfacePoint, py::return_value_policy::copy,
        R"pbdoc(
        Precompute logarithmic map for a given surface point.
    )pbdoc");

  m.def("vertexMap", &logmapVertex, py::return_value_policy::copy,
        R"pbdoc(
        Precompute logarithmic map for a given point.
    )pbdoc");

  m.def("surface_area", &surface_area, py::return_value_policy::copy, R"pbdoc(
        Computes surface area of the given mesh.
    )pbdoc");

  m.def("weights", &weights, py::return_value_policy::copy, R"pbdoc(
        Computes vertex lumped mass matrix for sampled points.
    )pbdoc");

  m.def("nearest", &nearest, py::return_value_policy::copy, R"pbdoc(
        Returns a mapping from all vertices to the nearest sampled points.
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
