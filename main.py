import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import csv

matplotlib.use("TkAgg")

# Nombres de las columnas del dataset wdbc.data
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float):
        """Inicializa la red neuronal con pesos y biases aleatorios."""
        self.learning_rate = learning_rate
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Función de activación sigmoide."""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivada de la función sigmoide."""
        return x * (1 - x)
    
    def forward(self, X: np.ndarray) -> tuple:
        """Propagación hacia adelante."""
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """Propagación hacia atrás para actualizar pesos y biases."""
        self.error = y - output
        self.delta2 = self.error * self.sigmoid_derivative(output)
        
        self.error_hidden = np.dot(self.delta2, self.weights2.T)
        self.delta1 = self.error_hidden * self.sigmoid_derivative(self.a1)
        
        self.weights2 += self.learning_rate * np.dot(self.a1.T, self.delta2)
        self.bias2 += self.learning_rate * np.sum(self.delta2, axis=0, keepdims=True)
        self.weights1 += self.learning_rate * np.dot(X.T, self.delta1)
        self.bias1 += self.learning_rate * np.sum(self.delta1, axis=0, keepdims=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> list:
        """Entrena la red neuronal y devuelve el error por época."""
        errors = []
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            error = np.mean(np.square(y - output))
            errors.append(error)
        return errors

class BreastCancerApp:
    def __init__(self, root: tk.Tk):
        """Inicializa la aplicación con la interfaz gráfica."""
        self.root = root
        self.root.title("Clasificación de Cáncer de Mama")
        self.root.geometry("1200x800")
        
        # Cargar dataset manualmente
        self.load_data()
        
        # Escalar datos manualmente
        self.X_scaled = self.standard_scale(self.X)
        
        # Configuración inicial
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.nn = None
        self.errors = []
        
        # Crear la interfaz gráfica
        self.create_gui()
    
    def load_data(self) -> None:
        """Carga el dataset desde un archivo CSV."""
        try:
            X = []
            y = []
            with open("wdbc.data", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    # Primera columna: ID (ignorada)
                    # Segunda columna: Diagnóstico (M = maligno, B = benigno)
                    label = 1 if row[1] == "M" else 0
                    # Columnas 2 en adelante: características
                    features = [float(x) for x in row[2:]]
                    X.append(features)
                    y.append(label)
            
            self.X = np.array(X)
            self.y = np.array(y).reshape(-1, 1)
            self.feature_names = FEATURE_NAMES
        except FileNotFoundError:
            messagebox.showerror("Error", "No se encontró el archivo 'wdbc.data'. Asegúrate de descargarlo y colocarlo en el directorio del script.")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar los datos: {str(e)}")
            self.root.quit()
    
    def standard_scale(self, X: np.ndarray) -> np.ndarray:
        """Escala los datos manualmente (media = 0, desviación estándar = 1)."""
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Evitar división por cero
        return (X - mean) / std
    
    def train_test_split_manual(self, X: np.ndarray, y: np.ndarray, train_size: float) -> tuple:
        """Divide los datos en conjunto de entrenamiento y prueba manualmente."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_samples = int(n_samples * train_size)
        train_indices = indices[:train_samples]
        test_indices = indices[train_samples:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def create_gui(self) -> None:
        """Crea los elementos de la interfaz gráfica."""
        # Frame superior para controles
        control_frame = ttk.LabelFrame(self.root, text="Controles", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Selección de características para el scatter plot
        ttk.Label(control_frame, text="Característica X:").grid(row=0, column=0, padx=5, pady=5)
        self.feature_x = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
        self.feature_x.grid(row=0, column=1, padx=5, pady=5)
        self.feature_x.set(self.feature_names[0])
        
        ttk.Label(control_frame, text="Característica Y:").grid(row=0, column=2, padx=5, pady=5)
        self.feature_y = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
        self.feature_y.grid(row=0, column=3, padx=5, pady=5)
        self.feature_y.set(self.feature_names[1])
        
        ttk.Button(control_frame, text="Visualizar Datos", command=self.plot_data).grid(row=0, column=4, padx=5, pady=5)
        
        # Parámetros de entrenamiento
        ttk.Label(control_frame, text="Tasa de Aprendizaje (η):").grid(row=1, column=0, padx=5, pady=5)
        self.learning_rate = tk.DoubleVar(value=0.01)
        ttk.Entry(control_frame, textvariable=self.learning_rate).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Número de Épocas:").grid(row=1, column=2, padx=5, pady=5)
        self.epochs = tk.IntVar(value=1000)
        ttk.Entry(control_frame, textvariable=self.epochs).grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(control_frame, text="% Datos Entrenamiento:").grid(row=1, column=4, padx=5, pady=5)
        self.train_split = tk.DoubleVar(value=0.8)
        ttk.Entry(control_frame, textvariable=self.train_split).grid(row=1, column=5, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Entrenar Red Neuronal", command=self.train_network).grid(row=1, column=6, padx=5, pady=5)
        
        # Frame para gráficos
        self.plot_frame = ttk.LabelFrame(self.root, text="Resultados", padding=10)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Configurar gráficos
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def plot_data(self) -> None:
        """Muestra un scatter plot de los datos seleccionados."""
        try:
            feature_x_idx = list(self.feature_names).index(self.feature_x.get())
            feature_y_idx = list(self.feature_names).index(self.feature_y.get())
            
            self.ax1.clear()
            scatter = self.ax1.scatter(self.X_scaled[:, feature_x_idx], self.X_scaled[:, feature_y_idx], 
                                     c=self.y.flatten(), cmap="coolwarm", alpha=0.6)
            self.ax1.set_xlabel(self.feature_names[feature_x_idx])
            self.ax1.set_ylabel(self.feature_names[feature_y_idx])
            self.ax1.set_title("Datos: Cáncer de Mama")
            self.ax1.legend(handles=scatter.legend_elements()[0], labels=["Benigno", "Maligno"])
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo visualizar los datos: {str(e)}")
    
    def train_network(self) -> None:
        """Entrena la red neuronal y actualiza los gráficos."""
        try:
            # Obtener parámetros
            learning_rate = self.learning_rate.get()
            epochs = self.epochs.get()
            train_split = self.train_split.get()
            
            if not (0 < learning_rate < 1):
                raise ValueError("La tasa de aprendizaje debe estar entre 0 y 1.")
            if not (epochs > 0):
                raise ValueError("El número de épocas debe ser mayor que 0.")
            if not (0 < train_split < 1):
                raise ValueError("El porcentaje de entrenamiento debe estar entre 0 y 1.")
            
            # Dividir datos manualmente
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split_manual(
                self.X_scaled, self.y, train_split
            )
            
            # Seleccionar solo las dos características para la frontera de decisión
            feature_x_idx = list(self.feature_names).index(self.feature_x.get())
            feature_y_idx = list(self.feature_names).index(self.feature_y.get())
            self.X_train_2d = self.X_train[:, [feature_x_idx, feature_y_idx]]
            self.X_test_2d = self.X_test[:, [feature_x_idx, feature_y_idx]]
            
            # Inicializar red neuronal
            self.nn = NeuralNetwork(input_size=2, hidden_size=10, output_size=1, learning_rate=learning_rate)
            
            # Entrenar red neuronal
            self.errors = self.nn.train(self.X_train_2d, self.y_train, epochs)
            
            # Actualizar gráficos
            self.update_plots(feature_x_idx, feature_y_idx)
            
            # Calcular y mostrar exactitud
            predictions = (self.nn.forward(self.X_test_2d) > 0.5).astype(int)
            accuracy = np.mean(predictions == self.y_test)
            messagebox.showinfo("Resultado", f"Exactitud en datos de prueba: {accuracy:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo entrenar la red neuronal: {str(e)}")
    
    def update_plots(self, feature_x_idx: int, feature_y_idx: int) -> None:
        """Actualiza los gráficos de error, frontera de decisión y datos."""
        # Gráfico de error vs. épocas
        self.ax2.clear()
        self.ax2.plot(self.errors, label="Error Cuadrático Medio")
        self.ax2.set_xlabel("Época")
        self.ax2.set_ylabel("Error")
        self.ax2.set_title("Error vs. Épocas")
        self.ax2.legend()
        
        # Gráfico de la frontera de decisión
        self.ax3.clear()
        x_min, x_max = self.X_scaled[:, feature_x_idx].min() - 1, self.X_scaled[:, feature_x_idx].max() + 1
        y_min, y_max = self.X_scaled[:, feature_y_idx].min() - 1, self.X_scaled[:, feature_y_idx].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = self.nn.forward(grid).reshape(xx.shape)
        
        self.ax3.contourf(xx, yy, preds, levels=[0, 0.5, 1], alpha=0.3, cmap="coolwarm")
        scatter = self.ax3.scatter(self.X_test_2d[:, 0], self.X_test_2d[:, 1], c=self.y_test.flatten(), 
                                 cmap="coolwarm", alpha=0.6)
        self.ax3.set_xlabel(self.feature_names[feature_x_idx])
        self.ax3.set_ylabel(self.feature_names[feature_y_idx])
        self.ax3.set_title("Frontera de Decisión")
        self.ax3.legend(handles=scatter.legend_elements()[0], labels=["Benigno", "Maligno"])
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = BreastCancerApp(root)
    root.mainloop()