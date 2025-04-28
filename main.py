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
    "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2",
    "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3",
    "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
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
            errors.append(total_error)  
        return errors

class clasificacionCancer:
    def __init__(self, root: tk.Tk):
        """Inicializa la aplicación con la interfaz gráfica."""
        self.root = root
        self.root.title("Clasificación de Cáncer de Mama")
        self.root.geometry("1200x800")
        
        # Cargar dataset manualmente
        self.load_data()
        
        # Normalizar datos manualmente y guardar parámetros de normalización
        self.datosNormalizados, self.mins, self.maxs = self.normalizar(self.X)
        
        
        self.datosEntreno = None
        self.datosPrueba = None
        self.solucionEntreno = None
        self.solucionPrueba = None
        self.perceptron = None
        self.errors = []
        
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
            self.X = np.array(X)  # datos sin normalizr, originales
            self.y = np.array(y).reshape(-1, 1)
            self.feature_names = FEATURE_NAMES
        except FileNotFoundError:
            messagebox.showerror("Error", "No se encontró el archivo 'wdbc.data'. Asegúrate de descargarlo y colocarlo en el mismo directorio de la aplicacion")
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar los datos: {str(e)}")
            self.root.quit()
    
    def normalizar(self, X: np.ndarray) -> tuple:
        """Normaliza los datos al rango [0, 1] y devuelve parámetros de normalización."""
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        # Evitar división por cero
        ranges = maxs - mins
        ranges[ranges == 0] = 1  # Si el rango es 0, establecerlo a 1 para evitar división por cero
        xNormalizados = (X - mins) / ranges
        return xNormalizados, mins, maxs   
    
    def inverse_normalize(self, X_normalized: np.ndarray, feature_idx: int) -> np.ndarray:
        """Transforma los datos normalizados de vuelta al espacio original."""
        return X_normalized * (self.maxs[feature_idx] - self.mins[feature_idx]) + self.mins[feature_idx]
    
    def dividirDatos(self, X: np.ndarray, y: np.ndarray, tamanioEntreno: float) -> tuple:
        """Divide los datos en conjunto de entrenamiento y prueba manualmente."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_samples = int(n_samples * tamanioEntreno)
        entrenoIndices = indices[:train_samples]
        test_indices = indices[train_samples:]
        
        datosEntreno = X[entrenoIndices]
        datosPrueba = X[test_indices]
        solucionEntreno = y[entrenoIndices]
        solucionPrueba = y[test_indices]
        
        return datosEntreno, datosPrueba, solucionEntreno, solucionPrueba, entrenoIndices, test_indices
    
    def create_gui(self) -> None:
        """Crea los elementos de la interfaz gráfica."""
        control_frame = ttk.LabelFrame(self.root, text="Controles", padding=10)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(control_frame, text="Característica X:").grid(row=0, column=0, padx=5, pady=5)
        self.seleccion_x = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
        self.seleccion_x.grid(row=0, column=1, padx=5, pady=5)
        self.seleccion_x.set(self.feature_names[0])
        
        ttk.Label(control_frame, text="Característica Y:").grid(row=0, column=2, padx=5, pady=5)
        self.seleccion_y = ttk.Combobox(control_frame, values=self.feature_names, state="readonly")
        self.seleccion_y.grid(row=0, column=3, padx=5, pady=5)
        self.seleccion_y.set(self.feature_names[1])
        
        ttk.Button(control_frame, text="Visualizar Datos", command=self.mostrarDatos).grid(row=0, column=4, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Tasa de Aprendizaje (η):").grid(row=1, column=0, padx=5, pady=5)
        self.tasa = tk.DoubleVar(value=0.01)
        ttk.Entry(control_frame, textvariable=self.tasa).grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Número de Épocas:").grid(row=1, column=2, padx=5, pady=5)
        self.epocas = tk.IntVar(value=100)
        ttk.Entry(control_frame, textvariable=self.epocas).grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Label(control_frame, text="% Datos Entrenamiento:").grid(row=1, column=4, padx=5, pady=5)
        self.porcentaje = tk.DoubleVar(value=0.8)
        ttk.Entry(control_frame, textvariable=self.porcentaje).grid(row=1, column=5, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Entrenar Perceptrón", command=self.entrenar).grid(row=1, column=6, padx=5, pady=5)
        
        self.plot_frame = ttk.LabelFrame(self.root, text="Resultados", padding=10)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def mostrarDatos(self) -> None:
        """Muestra un scatter plot de los datos seleccionados sin normalizarlos"""
        try:
            feature_x_idx = list(self.feature_names).index(self.seleccion_x.get())
            feature_y_idx = list(self.feature_names).index(self.seleccion_y.get())
            
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
    
    def entrenar(self) -> None:
        """Entrena el perceptrón y actualiza los gráficos."""
        try:
            tasa = self.tasa.get()
            epocas = self.epocas.get()
            porcentaje = self.porcentaje.get()
            
            if not (0 < tasa < 1):
                raise ValueError("La tasa de aprendizaje debe estar entre 0 y 1.")
            if not (epocas > 0):
                raise ValueError("El número de épocas debe ser mayor que 0.")
            if not (0 < porcentaje < 1):
                raise ValueError("El porcentaje de entrenamiento debe estar entre 0 y 1.")
            
            # Dividir datos manualmente (usando datos normalizados para el entrenamiento)
            self.datosEntreno, self.datosPrueba, self.solucionEntreno, self.solucionPrueba, train_indices, test_indices = self.dividirDatos(
                self.datosNormalizados, self.y, porcentaje
            )
            
            # Seleccionar solo las dos características para el perceptrón (normalizadas)
            xSeleccionados = list(self.feature_names).index(self.seleccion_x.get())
            ySeleccionados = list(self.feature_names).index(self.seleccion_y.get())
            self.datosEntrenamiento = self.datosEntreno[:, [xSeleccionados, ySeleccionados]]
            self.datosPruebaSE = self.datosPrueba[:, [xSeleccionados, ySeleccionados]]
            
            # También guardar las versiones originales para los gráficos
            self.datosPruebaGrafica = self.X[test_indices][:, [xSeleccionados, ySeleccionados]]
            # Inicializar perceptrón
            self.perceptron = Perceptron(input_size=2, tasa=tasa)
            
            # Entrenar perceptrón (con datos normalizados)
            self.errors = self.perceptron.train(self.datosEntrenamiento, self.solucionEntreno, epocas)
            
            # Actualizar gráficos
            self.update_plots(xSeleccionados, ySeleccionados)
            
            # Calcular y mostrar exactitud
            predictions = np.array([self.perceptron.activacion(x) for x in self.datosPruebaSE])
            accuracy = np.mean(predictions == self.solucionPrueba.flatten())
            messagebox.showinfo("Resultado", f"Exactitud en datos de prueba: {accuracy:.2%}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo entrenar el perceptrón: {str(e)}")
    
    def update_plots(self, Xseleccionados: int, Yseleccionados: int) -> None:
        """Actualiza los gráficos de error, frontera de decisión y datos (en escala original)."""
        # Gráfico de error vs. épocas
        self.ax2.clear()
        self.ax2.plot(self.errors, label="Total Error vs Epoca")
        self.ax2.set_xlabel("Época")
        self.ax2.set_ylabel("Error Total")
        self.ax2.set_title("Error vs. Épocas")
        self.ax2.legend()
        
        # Gráfico de la frontera de decisión (transformada al espacio original)
        self.ax3.clear()
        
        x_min = self.X[:, Xseleccionados].min()-1 
        x_max = self.X[:, Xseleccionados].max()+1
        y_min = self.X[:, Yseleccionados].min()-1
        y_max = self.X[:, Yseleccionados].max()+1
        
        # Crear una malla en el espacio original
        xx_orig, yy_orig = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        # Transformar la malla al espacio normalizado para hacer predicciones
        xx_normalized = (xx_orig - self.mins[Xseleccionados]) / (self.maxs[Xseleccionados] - self.mins[Xseleccionados])
        yy_normalized = (yy_orig - self.mins[Yseleccionados]) / (self.maxs[Yseleccionados] - self.mins[Yseleccionados])
        grid_normalized = np.c_[xx_normalized.ravel(), yy_normalized.ravel()]

        # Hacer predicciones con el perceptrón (en el espacio normalizado)
        preds = np.array([self.perceptron.activacion(x) for x in grid_normalized]).reshape(xx_orig.shape)
        
        # Graficar la frontera de decisión con datos sin normalizar
        self.ax3.contourf(xx_orig, yy_orig, preds, levels=[0, 0.5, 1], alpha=0.3, cmap="coolwarm")
        
        # Graficar los datos de prueba en el espacio original
        scatter = self.ax3.scatter(self.datosPruebaGrafica[:, 0], self.datosPruebaGrafica[:, 1], 
                                 c=self.solucionPrueba.flatten(), cmap="coolwarm", alpha=0.6)
        self.ax3.set_xlabel(self.feature_names[Xseleccionados])
        self.ax3.set_ylabel(self.feature_names[Yseleccionados])
        self.ax3.set_title("Frontera de Decisión")
        self.ax3.legend(handles=scatter.legend_elements()[0], labels=["Benigno", "Maligno"])
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = clasificacionCancer(root)
    root.mainloop()