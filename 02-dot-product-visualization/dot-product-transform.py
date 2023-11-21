import time
import numpy as np
import config as cfg
import streamlit as st
import plotly.graph_objs as go

class Grid():

    def __init__(self) -> None:
        pass

    def plot_basis(self, basis: np.ndarray, fig: go.Figure, vector: np.ndarray=None) -> go.Figure:
        annotation = []
        for index, bais_vector in enumerate(basis):
            annotation.append(
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
        if vector is not None:
            annotation.append(
                go.layout.Annotation(
                    {
                        "x": vector[0],
                        "y": vector[1],
                        "xref": "x",
                        "yref": "y",
                        "axref": "x",
                        "ayref": "y",
                        "ax": 0,
                        "ay": 0,
                        "arrowhead": 5,
                        "arrowwidth": 3,
                        "arrowcolor": cfg.VECTOR_COLOR
                    }
                )
            )
        fig.update_layout(annotations=annotation)
        return fig
        
    def plot_grid(self, basis: np.ndarray, grid: np.ndarray, name: str, extra: list[float, np.ndarray]=None) -> go.Figure:
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

        if name.split("-")[0] == "00":
            plot_title = "Initial Grid"
        elif name.split("-")[0] == "01":
            plot_title = "Transformed Grid"
        elif name.split("-")[0] == "02":
            plot_title = "Transformed Grid in Original Grid with random incline"
        else:
            plot_title = f"Dot of [1, 2]^T and [2, 1]^T = {extra[0]}"

        fig.update_layout(
            title=plot_title,
            xaxis_title="X axis",
            yaxis_title="Y axis",
            showlegend=False,
            template="plotly_dark",
            yaxis={"range": [-cfg.GRID_SIZE, cfg.GRID_SIZE]},
            xaxis={"range": [-cfg.GRID_SIZE, cfg.GRID_SIZE]}
        )
        if extra is not None:
            fig = self.plot_basis(basis, fig, extra[1])
        else:
            fig = self.plot_basis(basis, fig)
        fig.write_html(name)
        return fig
    
class InitialBasis():

    def __init__(self) -> None:
        """
        --> A set of 2 basis vectors are represented using the default unit vectors i_hat and j_hat.
        --> Hence the span is the default XY 2D plane, as the two are linerly independent.
        --> The i_hat and j_hat are [1, 0]^T and [0, 1]^T respectivly.
        --> Hence the matrix will be [[1, 0], [0, 1]].
        """
        # basis vectors are represented in a list one after the other for ease of coding.
        # here it looks same as the basis matrix.
        self.basis = cfg.INITIAL_BASIS
        self.grid_cords = np.arange(-cfg.GRID_SIZE, cfg.GRID_SIZE+1, cfg.GRID_SPACING)
        self.grid = np.asarray(
            [[x_cord, y_cord] for y_cord in self.grid_cords for x_cord in self.grid_cords]
            ).reshape((2*cfg.GRID_SIZE+1, 2*cfg.GRID_SIZE+1, 2))
        
class TransformBasis():

    def __init__(self) -> None:
        """
        --> New basis vectors are introduced such that new_i_hat is [1] and new_j_hat is [2].
        --> Certain Linear Transformation was resposible for these new basis vectors which can be ignored for now.
        --> Since the two new basis vectors are linearly dependent, the span is a 1D line.
        --> The matrix with these basis vectors is [1, 2].
        --> This [1, 2] matrix will represent a vector [[1], [2]]^T to be taken a dot with some another vector.
        """
        # basis vectors are represented in a list one after the other for easy of coding.
        # now to make the 2D grid, the transformation matrix was changed to [[1, 0], [2, 0]], so that all y cordinated will be zero (2D --> 1D).
        # here the transpose will be the correct matrix representation of these basis vectors (see line 108).
        self.basis = cfg.TRANSFORM_BASIS

    def perform_transformation(self, vector: np.ndarray) -> np.ndarray:
        return np.matmul(np.transpose(self.basis), vector)
    
class TranformBasisAsOriginal():

    def __init__(self) -> None:
        """
        --> The Transformation is represented in the original 2D span as a vector.
        --> Lets just put this 1D grid line from the transfomation using matrix transform onto a line in 2D space.
        --> The following transformation matrix can be used [[1, 2], [0.5, 1]] (simply adding the second dimension for the sake of plotting a 2D grid).
        --> the second row determines the incline of the plotted line.
        --> The new basis vectors will lie in the span of the matrix transform which is a 1D line passing through origin and (1, 2).
        --> Essentially, the transformed grid is put on this line.
        """
        # basis vectors are represented in a list one after the other for easy of coding.
        # here the transpose will be the correct matrix representation of these basis vectors (see line 124).
        self.basis = cfg.TRANSFORM_BASIS_AS_ORIGINAL

    def place_in_2dgrid(self, vector: np.ndarray) -> np.ndarray:
        return np.matmul(np.transpose(self.basis), vector)

    
if __name__ == "__main__":

    grid_obj = Grid()
    initial_basis_obj = InitialBasis()
    transformed_basis_obj = TransformBasis()
    tranformed_basis_as_original_obj = TranformBasisAsOriginal()


    # ========================================================================================================
    # ORIGINAL GRID
    # ========================================================================================================

    # ploting initial basis
    initial_grid_fig = grid_obj.plot_grid(initial_basis_obj.basis, initial_basis_obj.grid, "00-initial_basis.html")


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
    transformed_grid_fig = grid_obj.plot_grid(transformed_basis_obj.basis, transformed_grid, "01-transformed_grid.html")


    # ========================================================================================================
    # TRANSFORMED GRID IN ORIGINAL 2D GRID
    # ========================================================================================================

    # calculating transformation to place it on a line in original grid
    replaced_cords = []
    for line_cords in initial_basis_obj.grid:
        for cords in line_cords:
            replaced_cords.append(
                tranformed_basis_as_original_obj.place_in_2dgrid(
                    cords.reshape((-1, 1))
                ).reshape(1, -1)
            )
    replaced_grid = np.asarray(replaced_cords).reshape((2*cfg.GRID_SIZE+1, 2*cfg.GRID_SIZE+1, 2))

    # plotting the new basis
    replaced_grid_fig = grid_obj.plot_grid(tranformed_basis_as_original_obj.basis, replaced_grid, "02-replaced_grid.html")


    # ========================================================================================================
    # TRANSFORM 2D VECTOR ONTO THE GRID
    # ========================================================================================================

    # defining the vector
    vector = np.array([2, 1])

    # calculating dot
    dot_product = np.dot(vector, cfg.INITIAL_VECTOR)

    # calculating transformation to place it on a line in original grid
    transformed_vector = tranformed_basis_as_original_obj.place_in_2dgrid(vector.reshape(-1, 1)).reshape(-1)

    extras = [dot_product, transformed_vector]

    # plotting the new basis
    replaced_vector_grid_fig = grid_obj.plot_grid(tranformed_basis_as_original_obj.basis, replaced_grid, "03-dot_and_transform.html", extras)


    # ========================================================================================================
    
    all_figs = [initial_grid_fig, transformed_grid_fig, replaced_grid_fig, replaced_vector_grid_fig]

    placeholder = st.empty()
    while True:
        for fig in all_figs:
            with placeholder.container():
                st.plotly_chart(fig)
                time.sleep(3)
