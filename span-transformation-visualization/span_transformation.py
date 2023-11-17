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
        self.basis = np.array([[1, 0], [0, 1]])
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
        self.basis = np.array([[1, -2], [3, 0]])

    def perform_transformation(self, vector: np.ndarray) -> np.ndarray:
        return np.matmul(np.transpose(self.basis), vector)

if __name__ == "__main__":

    grid_obj = Grid()
    initial_basis_obj = InitialBasis()
    transformed_basis_obj = TransformedBasis()

    grid_obj.plot_grid(initial_basis_obj.basis, initial_basis_obj.grid, "initial_basis.html")

    transformed_cords = []
    for line_cords in initial_basis_obj.grid:
        for cords in line_cords:
            transformed_cords.append(
                transformed_basis_obj.perform_transformation(
                    cords.reshape((-1, 1))
                ).reshape(1, -1)
            )
    transformed_grid = np.asarray(transformed_cords).reshape((2*cfg.GRID_SIZE+1, 2*cfg.GRID_SIZE+1, 2))

    grid_obj.plot_grid(transformed_basis_obj.basis, transformed_grid, "transformed_grid.html")
