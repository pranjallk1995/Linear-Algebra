import numpy as np
import config as cfg
import plotly.graph_objs as go

class Grid():

    def __init__(self) -> None:
        pass

    def plot_basis(self, basis: np.ndarray, fig: go.Figure) -> go.Figure:
        basis_annotation = []
        for index, bais_vector in enumerate(basis):
            basis_annotation.append(
                go.layout.Annotation(
                    {
                        "x": bais_vector[0],
                        "y": bais_vector[1],
                        "xref": "x",
                        "yref": "y",
                        "axref": "x",
                        "ayref": "y",
                        "ax": 0,
                        "ay": 0,
                        "arrowhead": 5,
                        "arrowwidth": 3,
                        "arrowcolor": cfg.BASIS_COLORS[index]
                    }
                )
            )
        fig.update_layout(annotations=basis_annotation)
        return fig
        
    def plot_grid(self, basis: np.ndarray, grid: np.ndarray, name: str) -> go.Figure:
        x_traces = []
        y_traces = []
        for index in range(1, grid.shape[0]-1, 1):
            x_traces.append(
                go.Scatter(
                    x=grid[index, :, 0], y=grid[index, :, 1], mode="lines+markers",
                    hovertemplate="(%{x}, %{y})<extra></extra>", marker_color=cfg.GRID_COLOR
                )
            )
        for index in range(1, grid.shape[1]-1, 1):
            y_traces.append(
                go.Scatter(
                    x=grid[:, index, 0], y=grid[:, index, 1], mode="lines+markers",
                    hovertemplate="(%{x}, %{y})<extra></extra>", marker_color=cfg.GRID_COLOR
                )
            )
        fig = go.Figure(x_traces + y_traces)
        fig.update_layout(
            title=name.split(".")[0].split("_")[0].upper(),
            xaxis_title="X axis",
            yaxis_title="Y axis",
            showlegend=False,
            template="plotly_dark",
            yaxis={"range": [-cfg.GRID_SIZE, cfg.GRID_SIZE]},
            xaxis={"range": [-cfg.GRID_SIZE, cfg.GRID_SIZE]}
        )
        fig = self.plot_basis(basis, fig)
        fig.write_html(name)

class InitialBasis():

    def __init__(self) -> None:
        """
        --> A set of 2 basis vectors are represented using the default unit vectors i_hat and j_hat.
        --> Hence the span is the default XY 2D plane, as the two are linerly independent.
        --> The i_hat and j_hat are [1, 0]^T and [0, 1]^T respectivly.
        --> Hence the matrix will be [[1, 0], [0, 1]].
        """
        # basis vectors are represented in a list one after the other for easy of coding.
        # here it looks same as the basis matrix.
        self.basis = cfg.INITIAL_BASIS
        self.grid_cords = np.arange(-cfg.GRID_SIZE, cfg.GRID_SIZE+1, cfg.GRID_SPACING)
        self.grid = np.asarray(
            [[x_cord, y_cord] for y_cord in self.grid_cords for x_cord in self.grid_cords]
            ).reshape((2*cfg.GRID_SIZE+1, 2*cfg.GRID_SIZE+1, 2))
        
class TransformedBasis():

    def __init__(self) -> None:
        """
        --> New basis vectors are introduced such that new_i_hat is [1, -2]^T and new_j_hat is [3, 0]^T.
        --> Certain Linear Transformation was resposible for these new basis vectors which can be ignored for now.
        --> Since the two new basis vectors are linearly independent, the span is a 2D plane again.
        --> The matrix with these basis vectors is [[1, 3], [-2, 0]].
        """
        # basis vectors are represented in a list one after the other for easy of coding.
        # here is the transpose will be the correct matrix representation of these basis vectors (see line 91).
        self.basis = cfg.TRANSFORMED_BASIS

    def perform_transformation(self, vector: np.ndarray) -> np.ndarray:
        return np.matmul(np.transpose(self.basis), vector)
    
class InverseTransformBasis():

    def __init__(self, original_basis: np.ndarray) -> None:
        """
        --> The inverse of a matrix results in the inverse transform of the original matrix.
        --> Example: if transformation was 90 degree clockwise rotation, the inverse transform will be 90 degree anti-clockwise rotation.
        --> i.e. 90 degree rotation matrix = [[0, 1], [-1, 0]], 90 degree anti-clockwise = [[0, -1], [1, 0]].
        --> invesre of one matrix above gives the other.
        --> [[0, 1], [-1, 0]]^-1 = [[0, -1], [1, 0]] and vice versa.
        """
        # takes in vectors, one after the other as before, not in matrix form.
        # stores then in the vector after vector format as well, hence the two transpose.
        self.basis = np.transpose(np.linalg.inv(np.transpose(original_basis)))

    def perform_inverse_transformation(self, vector: np.ndarray) -> np.ndarray:
        return np.matmul(np.transpose(self.basis), vector)

if __name__ == "__main__":

    grid_obj = Grid()
    initial_basis_obj = InitialBasis()
    transformed_basis_obj = TransformedBasis()
    inverse_transform_basis_obj = InverseTransformBasis(transformed_basis_obj.basis)


    # ========================================================================================================
    # ORIGINAL GRID
    # ========================================================================================================

    # ploting initial basis
    grid_obj.plot_grid(initial_basis_obj.basis, initial_basis_obj.grid, "00-initial_basis.html")


    # ========================================================================================================
    # TRANSFORMED GRID
    # ========================================================================================================

    # calculating transformation by the new basis
    transformed_cords = []
    for line_cords in initial_basis_obj.grid:
        for cords in line_cords:
            transformed_cords.append(
                transformed_basis_obj.perform_transformation(
                    cords.reshape((-1, 1))
                ).reshape(1, -1)
            )
    transformed_grid = np.asarray(transformed_cords).reshape((2*cfg.GRID_SIZE+1, 2*cfg.GRID_SIZE+1, 2))

    # plotting the new basis
    grid_obj.plot_grid(transformed_basis_obj.basis, transformed_grid, "01-transformed_grid.html")


    # ========================================================================================================
    # INVERSE TRANSFORMATION GRID
    # ========================================================================================================

    # calculating the inverse transformation grid
    inverse_transform_cords = []
    for line_cords in initial_basis_obj.grid:
        for cords in line_cords:
            inverse_transform_cords.append(
                inverse_transform_basis_obj.perform_inverse_transformation(
                    cords.reshape((-1, 1))
                ).reshape(1, -1)
            )
    inverse_transform_grid = np.asarray(inverse_transform_cords).reshape((2*cfg.GRID_SIZE+1, 2*cfg.GRID_SIZE+1, 2))

    # plotting the inverse transform matrix
    grid_obj.plot_grid(inverse_transform_basis_obj.basis, inverse_transform_grid, "02-inverse_transform_grid.html")


    # ========================================================================================================
    # INVERSE TRANSFORMED GRID
    # ========================================================================================================

    # calculating inverse transformation from the new basis to original basis
    inverse_transformed_cords = []
    for line_cords in transformed_grid:
        for cords in line_cords:
            inverse_transformed_cords.append(
                inverse_transform_basis_obj.perform_inverse_transformation(
                    cords.reshape((-1, 1))
                ).reshape(1, -1)
            )
    inverse_transformed_grid = np.asarray(inverse_transformed_cords).reshape((2*cfg.GRID_SIZE+1, 2*cfg.GRID_SIZE+1, 2))

    # calculating the inverse of the transformed basis
    inverse_transformed_basis = np.transpose(
        np.matmul(np.transpose(inverse_transform_basis_obj.basis), np.transpose(transformed_basis_obj.basis))
    )

    # plotting the inverted basis
    grid_obj.plot_grid(inverse_transformed_basis, inverse_transformed_grid, "03-inverse_transformed_grid.html")
