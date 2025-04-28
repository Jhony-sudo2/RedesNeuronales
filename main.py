import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import csv

matplotlib.use("TkAgg")

# Nombres de las columnas del dataset (basado en la documentación del UCI)
FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

class Perceptron:
    def __init__(self, input_size, tasa):
        """Inicializa el perceptrón con pesos y sesgo aleatorios."""
        self.tasa = tasa
        self.pesos = np.random.uniform(-1, 1, size=input_size)
        self.sesgo = np.random.uniform(-1, 1)

    def activacion(self, x):
        """Calcula la activación del perceptrón."""
        z = self.pesos * x
        if z.sum() + self.sesgo > 0:
            return 1
        else:
            return 0

    def train(self, X, y, epocas):
        """Entrena el perceptrón y devuelve una lista vacía de errores."""
        errors = []
        for epoca in range(epocas):
            total_error = 0
            for i in range(X.shape[0]):   
                prediccion = self.activacion(X[i])
                error = y[i] - prediccion 
                total_error += error ** 2
                self.pesos[0] += self.tasa * X[i,0].item() * error
                self.pesos[1] += self.tasa * X[i,1].item() * error
                self.sesgo += self.tasa * error
            print("Error: ",total_error)
            errors.append(total_error)  # Acumular el error por época
        return errors

class BreastCancerApp:
    def __init__(self, root: tk.Tk):
        """Inicializa la aplicación con la interfaz gráfica."""
        self.root = root
        self.root.title("Clasificación de Cáncer de Mama (Perceptrón)")
        self.root.geometry("1200x800")
        
        # Cargar dataset manualmente
        self.load_data()
        
        # Normalizar datos manualmente y guardar parámetros de normalización
        self.X_normalized, self.mins, self.maxs = self.normalize(self.X)
        
        # Configuración inicial
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.perceptron = None
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
                    label = 1 if row[1] == "M" else 0
                    features = [float(x) for x in row[2:]]
                    X.append(features)
                    y.append(label)
                    # Imprimir la primera muestra con nombres de características para verificar el orden
                    if len(X) == 1:
                        print("Primera muestra cargada:")
                        for name, value in zip(FEATURE_NAMES, features):
                            print(f"{name}: {value}")
        
            self.X = np.array(X)  # Datos originales
            self.y = np.array(y).reshape(-1, 1)
            self.feature_names = FEATURE_NAMES
        except FileNotFoundError:
            messagebox.showerror("Error", "No se encontró el archivo 'wdbc.data'. Asegúrate de descargarlo y colocarlo en el directorio del script.")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar los datos: {str(e)}")
            self.root.quit()
    
    def normalize(self, X: np.ndarray) -> tuple:
        """Normaliza los datos al rango [0, 1] y devuelve parámetros de normalización."""
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        # Evitar división por cero
        ranges = maxs - mins
        ranges[ranges == 0] = 1  # Si el rango es 0, establecerlo a 1 para evitar división por cero
        X_normalized = (X - mins) / ranges
        return X_normalized, mins, maxs
    
    def inverse_normalize(self, X_normalized: np.ndarray, feature_idx: int) -> np.ndarray:
        """Transforma los datos normalizados de vuelta al espacio original."""
        return X_normalized * (self.maxs[feature_idx] - self.mins[feature_idx]) + self.mins[feature_idx]
    
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
        
        return X_train, X_test, y_train, y_test, train_indices, test_indices
    
    def create_gui(self) -> None:
        """Crea los elementos de la interfaz gráfica."""
        control_frame = ttk.LabelFrame(self.root, text="Controles", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(control_frame, text="Característica X:").grid(row=0, column=0, padx=5, pady=5)
        self.feature_x = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
        self.feature_x.grid(row=0, column=1, padx=5, pady=5)
        self.feature_x.set(self.feature_names[0])
        
        ttk.Label(control_frame, text="Característica Y:").grid(row=0, column=2, padx=5, pady=5)
        self.feature_y = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
        self.feature_y.grid(row=0, column=3, padx=5, pady=5)
        self.feature_y.set(self.feature_names[1])
        
        ttk.Button(control_frame, text="Visualizar Datos", command=self.plot_data).grid(row=0, column=4, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Tasa de Aprendizaje (η):").grid(row=1, column=0, padx=5, pady=5)
        self.learning_rate = tk.DoubleVar(value=0.01)
        ttk.Entry(control_frame, textvariable=self.learning_rate).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Número de Épocas:").grid(row=1, column=2, padx=5, pady=5)
        self.epochs = tk.IntVar(value=100)
        ttk.Entry(control_frame, textvariable=self.epochs).grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(control_frame, text="% Datos Entrenamiento:").grid(row=1, column=4, padx=5, pady=5)
        self.train_split = tk.DoubleVar(value=0.8)
        ttk.Entry(control_frame, textvariable=self.train_split).grid(row=1, column=5, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Entrenar Perceptrón", command=self.train_perceptron).grid(row=1, column=6, padx=5, pady=5)
        
        self.plot_frame = ttk.LabelFrame(self.root, text="Resultados", padding=10)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def plot_data(self) -> None:
        """Muestra un scatter plot de los datos seleccionados (en escala original)."""
        try:
            feature_x_idx = list(self.feature_names).index(self.feature_x.get())
            feature_y_idx = list(self.feature_names).index(self.feature_y.get())
            
            self.ax1.clear()
            scatter = self.ax1.scatter(self.X[:, feature_x_idx], self.X[:, feature_y_idx], 
                                     c=self.y.flatten(), cmap="coolwarm", alpha=0.6)
            self.ax1.set_xlabel(self.feature_names[feature_x_idx])
            self.ax1.set_ylabel(self.feature_names[feature_y_idx])
            self.ax1.set_title("Datos: Cáncer de Mama")
            self.ax1.legend(handles=scatter.legend_elements()[0], labels=["Benigno", "Maligno"])
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo visualizar los datos: {str(e)}")
    
    def train_perceptron(self) -> None:
        """Entrena el perceptrón y actualiza los gráficos."""
        try:
            tasa = self.learning_rate.get()
            epocas = self.epochs.get()
            train_split = self.train_split.get()
            
            if not (0 < tasa < 1):
                raise ValueError("La tasa de aprendizaje debe estar entre 0 y 1.")
            if not (epocas > 0):
                raise ValueError("El número de épocas debe ser mayor que 0.")
            if not (0 < train_split < 1):
                raise ValueError("El porcentaje de entrenamiento debe estar entre 0 y 1.")
            
            # Dividir datos manualmente (usando datos normalizados para el entrenamiento)
            self.X_train, self.X_test, self.y_train, self.y_test, train_indices, test_indices = self.train_test_split_manual(
                self.X_normalized, self.y, train_split
            )
            
            # Seleccionar solo las dos características para el perceptrón (normalizadas)
            feature_x_idx = list(self.feature_names).index(self.feature_x.get())
            feature_y_idx = list(self.feature_names).index(self.feature_y.get())
            print(f"Característica X seleccionada: {self.feature_x.get()} (índice {feature_x_idx})")
            print(f"Característica Y seleccionada: {self.feature_y.get()} (índice {feature_y_idx})")
            self.X_train_2d = self.X_train[:, [feature_x_idx, feature_y_idx]]
            self.X_test_2d = self.X_test[:, [feature_x_idx, feature_y_idx]]
            
            # También guardar las versiones originales para los gráficos
            self.X_train_2d_orig = self.X[train_indices][:, [feature_x_idx, feature_y_idx]]
            self.X_test_2d_orig = self.X[test_indices][:, [feature_x_idx, feature_y_idx]]
            
            # Inicializar perceptrón
            self.perceptron = Perceptron(input_size=2, tasa=tasa)
            
            # Entrenar perceptrón (con datos normalizados)
            self.errors = self.perceptron.train(self.X_train_2d, self.y_train, epocas)
            
            # Actualizar gráficos
            self.update_plots(feature_x_idx, feature_y_idx)
            
            # Calcular y mostrar exactitud
            predictions = np.array([self.perceptron.activacion(x) for x in self.X_test_2d])
            accuracy = np.mean(predictions == self.y_test.flatten())
            messagebox.showinfo("Resultado", f"Exactitud en datos de prueba: {accuracy:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo entrenar el perceptrón: {str(e)}")
    
    def update_plots(self, feature_x_idx: int, feature_y_idx: int) -> None:
        """Actualiza los gráficos de error, frontera de decisión y datos (en escala original)."""
        # Gráfico de error vs. épocas
        self.ax2.clear()
        self.ax2.plot(self.errors, label="Error Cuadrático (Suma)")
        self.ax2.set_xlabel("Época")
        self.ax2.set_ylabel("Error Total")
        self.ax2.set_title("Error vs. Épocas")
        self.ax2.legend()
        
        # Gráfico de la frontera de decisión (transformada al espacio original)
        self.ax3.clear()
        
        # Definir límites en el espacio normalizado
        x_min_normalized = self.X_normalized[:, feature_x_idx].min() - 0.1
        x_max_normalized = self.X_normalized[:, feature_x_idx].max() + 0.1
        y_min_normalized = self.X_normalized[:, feature_y_idx].min() - 0.1
        y_max_normalized = self.X_normalized[:, feature_y_idx].max() + 0.1
        
        # Transformar los límites al espacio original
        x_min = self.inverse_normalize(np.array([x_min_normalized]), feature_x_idx)[0]
        x_max = self.inverse_normalize(np.array([x_max_normalized]), feature_x_idx)[0]
        y_min = self.inverse_normalize(np.array([y_min_normalized]), feature_y_idx)[0]
        y_max = self.inverse_normalize(np.array([y_max_normalized]), feature_y_idx)[0]
        
        # Crear una malla en el espacio original
        xx_orig, yy_orig = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        # Transformar la malla al espacio normalizado para hacer predicciones
        xx_normalized = (xx_orig - self.mins[feature_x_idx]) / (self.maxs[feature_x_idx] - self.mins[feature_x_idx])
        yy_normalized = (yy_orig - self.mins[feature_y_idx]) / (self.maxs[feature_y_idx] - self.mins[feature_y_idx])
        grid_normalized = np.c_[xx_normalized.ravel(), yy_normalized.ravel()]
        
        # Hacer predicciones con el perceptrón (en el espacio normalizado)
        preds = np.array([self.perceptron.activacion(x) for x in grid_normalized]).reshape(xx_orig.shape)
        
        # Graficar la frontera de decisión en el espacio original
        self.ax3.contourf(xx_orig, yy_orig, preds, levels=[0, 0.5, 1], alpha=0.3, cmap="coolwarm")
        
        # Graficar los datos de prueba en el espacio original
        scatter = self.ax3.scatter(self.X_test_2d_orig[:, 0], self.X_test_2d_orig[:, 1], 
                                 c=self.y_test.flatten(), cmap="coolwarm", alpha=0.6)
        self.ax3.set_xlabel(self.feature_names[feature_x_idx])
        self.ax3.set_ylabel(self.feature_names[feature_y_idx])
        self.ax3.set_title("Frontera de Decisión")
        self.ax3.legend(handles=scatter.legend_elements()[0], labels=["Benigno", "Maligno"])
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = BreastCancerApp(root)
    root.mainloop()