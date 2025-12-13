import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import difflib

# Configuración visual
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 10)
plt.rcParams['font.size'] = 11

class F1AnalyzerGUI:
    def __init__(self, root):
        """Inicializa la interfaz gráfica mejorada."""
        self.root = root
        self.root.title("F1 Analyzer Pro - Machine Learning Edition")
        self.root.geometry("1600x950")
        self.root.configure(bg="#0a0a0a")
        # Maximizar ventana
        try:
            self.root.state('zoomed')  # Windows
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                pass
        # Variables de datos
        self.data_path = os.path.join(os.getcwd(), "Data")
        self.drivers = pd.DataFrame()
        self.races = pd.DataFrame()
        self.results = pd.DataFrame()
        self.constructors = pd.DataFrame()
        self.lap_times = pd.DataFrame()
        self.df_clean = pd.DataFrame()
        # Variables del modelo Random Forest
        self.rf_model = None
        self.temporada = pd.DataFrame()
        self.datos_2024 = pd.DataFrame()
        self.model_metrics = {}
        # Cargar datos y entrenar modelo
        self.load_data()
        self.train_random_forest_model()
        # Crear interfaz
        self.create_widgets()

    # ============================================================
    # CARGA DE DATOS
    # ============================================================
    def load_data(self):
        """Carga todos los datasets necesarios."""
        files = {
            'drivers': "drivers.csv",
            'races': "races.csv",
            'results': "results.csv",
            'lap_times': "lap_times.csv",
            'constructors': "constructors.csv",
            'df_clean': "f1_clean_dataset.csv"
        }
        for attr, fname in files.items():
            path = os.path.join(self.data_path, fname)
            try:
                df = pd.read_csv(path, na_values=['\\N', 'NULL', '', 'nan'])
                setattr(self, attr, df)
            except FileNotFoundError:
                messagebox.showwarning(
                    "Archivo no encontrado",
                    f"No se encontró '{fname}'. Algunas funciones estarán limitadas."
                )
                setattr(self, attr, pd.DataFrame())
            except Exception as ex:
                messagebox.showerror("Error", f"Error leyendo {fname}:\n{ex}")
                setattr(self, attr, pd.DataFrame())

    # ============================================================
    # MODELO RANDOM FOREST
    # ============================================================
    def train_random_forest_model(self):
        """Entrena el modelo Random Forest para predicción 2025."""
        if self.df_clean.empty:
            self.rf_model = None
            return
        try:
            df = self.df_clean.copy()
            df.rename(columns={
                'points': 'puntos',
                'year': 'año',
                'team_name': 'constructor',
                'driver_fullname': 'nombre_piloto'
            }, inplace=True)
            df[['puntos', 'año']] = df[['puntos', 'año']].apply(pd.to_numeric, errors='coerce').fillna(0)
            df['nombre_piloto'] = df['nombre_piloto'].astype(str).str.strip()
            df['constructor'] = df['constructor'].astype(str).str.strip()

            puntos_piloto = df.groupby(['año', 'nombre_piloto'])['puntos'].sum().reset_index()
            puntos_piloto['posicion'] = puntos_piloto.groupby('año')['puntos'].rank(ascending=False, method='min')
            equipo_piloto = df.groupby(['año', 'nombre_piloto'])['constructor'].agg(
                lambda x: x.mode().iat[0] if not x.mode().empty else None
            ).reset_index()
            temporada = pd.merge(puntos_piloto, equipo_piloto, on=['año', 'nombre_piloto'])

            puntos_const = df.groupby(['año', 'constructor'])['puntos'].sum().reset_index()
            puntos_const['rank_team'] = puntos_const.groupby('año')['puntos'].rank(ascending=False, method='min')
            temporada = pd.merge(temporada, puntos_const[['año', 'constructor', 'rank_team']], on=['año', 'constructor'], how='left')
            self.temporada = temporada

            t_actual = temporada.copy()
            t_prev = temporada.copy()
            t_prev['año'] += 1
            t_prev.rename(columns={'puntos': 'pts_ant', 'posicion': 'pos_ant', 'rank_team': 'rank_team_ant'}, inplace=True)
            datos = pd.merge(t_actual, t_prev[['año', 'nombre_piloto', 'pts_ant', 'pos_ant', 'rank_team_ant']],
                             on=['año', 'nombre_piloto'], how='inner')
            datos = datos.dropna(subset=['pts_ant', 'pos_ant', 'rank_team_ant'])

            ultimo_anio = datos['año'].max()
            train = datos[datos['año'] < ultimo_anio]
            test = datos[datos['año'] == ultimo_anio]
            X_train = train[['pts_ant', 'pos_ant', 'rank_team_ant']]
            y_train = train['posicion']
            X_test = test[['pts_ant', 'pos_ant', 'rank_team_ant']]
            y_test = test['posicion']

            self.rf_model = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=90, n_jobs=-1)
            self.rf_model.fit(X_train, y_train)
            y_pred = self.rf_model.predict(X_test)

            self.model_metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'test_year': ultimo_anio
            }

            X = datos[['pts_ant', 'pos_ant', 'rank_team_ant']]
            y = datos['posicion']
            self.rf_model.fit(X, y)

            datos_2024 = temporada[temporada['año'] == 2024].copy()
            datos_2024.rename(columns={'puntos': 'pts_ant', 'posicion': 'pos_ant', 'rank_team': 'rank_team_ant'}, inplace=True)
            datos_2024['prediccion_2025'] = self.rf_model.predict(datos_2024[['pts_ant', 'pos_ant', 'rank_team_ant']])
            self.datos_2024 = datos_2024.sort_values('prediccion_2025').reset_index(drop=True)
        except Exception as ex:
            messagebox.showerror("Error", f"Error entrenando modelo Random Forest:\n{ex}")
            self.rf_model = None

    # ============================================================
    # INTERFAZ DE USUARIO
    # ============================================================
    def create_widgets(self):
        header_frame = tk.Frame(self.root, bg="#e10600", height=90)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        tk.Label(header_frame, text="🏎️ F1 ANALYZER PRO", font=("Arial", 30, "bold"), bg="#e10600", fg="white").pack(pady=8)
        tk.Label(header_frame, text="Machine Learning Edition • Random Forest Predictions", font=("Arial", 13), bg="#e10600", fg="#ffeeee").pack()

        left_panel = tk.Frame(self.root, bg="#1a1a1a", width=340)
        left_panel.pack(fill=tk.Y, side=tk.LEFT, padx=2, pady=2)
        left_panel.pack_propagate(False)

        canvas = tk.Canvas(left_panel, bg="#1a1a1a", highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#1a1a1a")
        scrollable_frame.bind("<Configure>", lambda el: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.create_menu_section(scrollable_frame, "🏁 PILOTOS", [
            ("Top Victorias", self.show_top_winners),
            ("Top Podios", self.show_top_podiums),
            ("Top Puntos", self.show_top_points),
            ("🔍 Buscar Piloto", self.advanced_search_driver)
        ])
        self.create_menu_section(scrollable_frame, "🏆 EQUIPOS", [
            ("Top Victorias", self.show_top_constructor_wins),
            ("Top Puntos", self.show_top_constructor_points),
            ("Evolución Temporal", self.show_constructor_evolution)
        ])
        self.create_menu_section(scrollable_frame, "⚔️ COMPARACIONES", [
            ("Comparar Pilotos", self.compare_drivers)
        ])
        self.create_menu_section(scrollable_frame, "🤖 MACHINE LEARNING", [
            ("Predicción 2025 - Top 10", self.show_2025_predictions),
            ("Rendimiento del Modelo", self.show_rf_model_performance),
            ("Predecir Piloto 2025", self.predict_driver_2025)
        ], bg="#9333ea")
        self.create_menu_section(scrollable_frame, "📊 VISUALIZACIONES", [
            ("Mapa de Calor", self.show_heatmap),
            ("Distribución Posiciones", self.show_distribution),
            ("Evolución Top 5", self.show_top5_evolution)
        ])
        self.create_menu_section(scrollable_frame, "📈 ESTADÍSTICAS", [
            ("Resumen General", self.show_general_stats)
        ], bg="#00D2BE")

        self.main_area = tk.Frame(self.root, bg="#0a0a0a")
        self.main_area.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=2, pady=2)
        self.show_welcome_message()

    def create_menu_section(self, parent, title, buttons, bg="#e10600"):
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, padx=10, pady=8)
        tk.Label(parent, text=title, font=("Arial", 12, "bold"), bg="#1a1a1a", fg="white", anchor="w", padx=15, pady=8).pack(fill=tk.X)
        for text, command in buttons:
            self.create_styled_button(parent, text, command, bg)

    def create_styled_button(self, parent, text, command, bg="#e10600"):
        button = tk.Button(parent, text=text, command=command, font=("Arial", 11), bg=bg, fg="white",
                           activebackground=self.get_hover_color(bg), activeforeground="white", relief=tk.FLAT,
                           cursor="hand2", padx=12, pady=12, anchor="w")
        button.pack(fill=tk.X, padx=12, pady=3)
        hover_color = self.get_hover_color(bg)
        button.bind("<Enter>", lambda er: button.config(bg=hover_color))
        button.bind("<Leave>", lambda er: button.config(bg=bg))

    def get_hover_color(self, base_color):
        colors = {"#e10600": "#ff2020", "#9333ea": "#a855f7", "#00D2BE": "#00e5d0"}
        return colors.get(base_color, "#ff2020")

    def clear_main_area(self):
        for widget in self.main_area.winfo_children():
            widget.destroy()

    def show_welcome_message(self):
        self.clear_main_area()
        frame = tk.Frame(self.main_area, bg="#0a0a0a")
        frame.pack(expand=True)
        tk.Label(frame, text="🏁 F1 ANALYZER PRO 🏁", font=("Arial", 36, "bold"), bg="#0a0a0a", fg="white").pack(pady=20)
        tk.Label(frame, text="Machine Learning Edition", font=("Arial", 20), bg="#0a0a0a", fg="#9333ea").pack(pady=5)
        tk.Label(frame, text="Random Forest Predictions • Advanced Analytics", font=("Arial", 14), bg="#0a0a0a", fg="#00D2BE").pack(pady=10)
        if self.rf_model:
            model_info = f"""
┌─────────────────────────────────────────────────────────────────────┐
│          MODELO RANDOM FOREST ACTIVO                                │
├─────────────────────────────────────────────────────────────────────┤
│  Algoritmo: Random Forest Regressor                                 │
│  Árboles: 300                                                       │
│  Error Medio Absoluto: {self.model_metrics.get('mae', 0):.2f}       │
│  RMSE: {self.model_metrics.get('rmse', 0):.2f}                      │
│  R² Score: {self.model_metrics.get('r2', 0):.3f}                    │
│  Año de validación: {int(self.model_metrics.get('test_year', 0))}   │
└─────────────────────────────────────────────────────────────────────┘
"""
        else:
            model_info = "\n⚠️  MODELO NO DISPONIBLE\n"
        tk.Label(frame, text=model_info, font=("Courier", 12), bg="#0a0a0a", fg="#00ff88", justify=tk.LEFT).pack(pady=20)
        dataset_info = f"""
📊 DATASET CARGADO:
• Pilotos: {len(self.drivers)}
• Carreras: {len(self.races)}
• Equipos: {len(self.constructors)}
• Período: {self.races['year'].min() if not self.races.empty else 'N/A'} - {self.races['year'].max() if not self.races.empty else 'N/A'}
• Predicciones 2025: {'✓ Disponibles' if not self.datos_2024.empty else '✗ No disponibles'}
"""
        tk.Label(frame, text=dataset_info, font=("Arial", 13), bg="#0a0a0a", fg="white", justify=tk.LEFT).pack(pady=20)
        tk.Label(frame, text="👈 Selecciona una opción del menú para comenzar", font=("Arial", 12), bg="#0a0a0a", fg="#888888").pack(pady=10)

    def show_plot_in_main_area(self, fig, title="Resultado"):
        self.clear_main_area()
        tk.Label(self.main_area, text=title, font=("Arial", 22, "bold"), bg="#0a0a0a", fg="white", pady=15).pack()
        canvas = FigureCanvasTkAgg(fig, master=self.main_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

    def show_text_result(self, text, title="Resultado"):
        self.clear_main_area()
        tk.Label(self.main_area, text=title, font=("Arial", 22, "bold"), bg="#0a0a0a", fg="white", pady=15).pack()
        text_area = scrolledtext.ScrolledText(self.main_area, font=("Courier", 12), bg="#1a1a1a", fg="white",
                                              insertbackground="white", wrap=tk.WORD, padx=30, pady=30)
        text_area.pack(fill=tk.BOTH, expand=True, padx=25, pady=15)
        text_area.insert(tk.END, text)
        text_area.config(state=tk.DISABLED)

    # ============================================================
    # BÚSQUEDA MEJORADA
    # ============================================================
    def advanced_search_driver(self):
        if self.drivers.empty or self.results.empty:
            messagebox.showwarning("Datos", "No hay datos para buscar.")
            return
        search_window = tk.Toplevel(self.root)
        search_window.title("🔍 Búsqueda Avanzada de Pilotos")
        search_window.geometry("1100x750")
        search_window.configure(bg="#1a1a1a")
        tk.Label(search_window, text="🔍 Búsqueda Inteligente de Pilotos", font=("Arial", 18, "bold"), bg="#1a1a1a", fg="white").pack(pady=15)
        tk.Label(search_window, text="Ingresa cualquier parte del nombre (ej: 'Max', 'Hamilton', 'Schum')", font=("Arial", 11), bg="#1a1a1a", fg="#888888").pack(pady=5)
        entry = tk.Entry(search_window, font=("Arial", 14), width=60)
        entry.pack(pady=12)
        entry.focus()

        results_frame = tk.Frame(search_window, bg="#1a1a1a")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=12)
        scrollbar = tk.Scrollbar(results_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        listbox = tk.Listbox(results_frame, font=("Courier", 12), bg="#2d2d2d", fg="white", selectbackground="#e10600", yscrollcommand=scrollbar.set, height=12)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        info_text = scrolledtext.ScrolledText(search_window, font=("Courier", 11), bg="#2d2d2d", fg="white", height=15, wrap=tk.WORD)
        info_text.pack(fill=tk.BOTH, padx=25, pady=12)
        matches_data = []

        def do_search():
            nonlocal matches_data
            search_term = entry.get().strip()
            if not search_term:
                messagebox.showwarning("Advertencia", "Ingresa un término")
                return
            matches = self.drivers[
                self.drivers['forename'].str.contains(search_term, case=False, na=False) |
                self.drivers['surname'].str.contains(search_term, case=False, na=False)
            ].copy()
            listbox.delete(0, tk.END)
            matches_data = []
            if matches.empty:
                listbox.insert(tk.END, f"  ❌ No se encontraron pilotos con '{search_term}'")
                info_text.delete(1.0, tk.END)
                return
            listbox.insert(tk.END, f"  ✓ Se encontraron {len(matches)} piloto(s):")
            listbox.insert(tk.END, "")
            for idx, driver in matches.iterrows():
                driver_name = f"{driver.get('forename', '')} {driver.get('surname', '')}"
                driver_id = driver['driverId']
                driver_results = self.results[self.results['driverId'] == driver_id]
                wins = len(driver_results[driver_results['positionOrder'] == 1])
                podiums = len(driver_results[driver_results['positionOrder'].isin([1, 2, 3])])
                points = driver_results['points'].sum()
                display_text = f"  {len(matches_data)+1}. {driver_name:<28} | {wins:3} 🏆 | {podiums:3} 🥇 | {points:7.0f} pts"
                listbox.insert(tk.END, display_text)
                matches_data.append((driver, driver_results))
            info_text.delete(1.0, tk.END)
            info_text.insert(tk.END, "💡 Haz clic en un piloto para ver estadísticas completas")
        def show_driver_details(event):
            selection = listbox.curselection()
            if not selection or selection[0] < 2:
                return
            idx = selection[0] - 2
            if idx < 0 or idx >= len(matches_data):
                return
            driver, driver_results = matches_data[idx]
            driver_name = f"{driver.get('forename', '')} {driver.get('surname', '')}"
            wins = len(driver_results[driver_results['positionOrder'] == 1])
            podiums = len(driver_results[driver_results['positionOrder'].isin([1, 2, 3])])
            points = driver_results['points'].sum()
            races = len(driver_results)
            avg_pos = driver_results['positionOrder'].mean() if races > 0 else 0
            details = f"""
{'='*75}
  {driver_name}
{'='*75}
📋 INFORMACIÓN PERSONAL:
  • Nacionalidad: {driver.get('nationality', 'N/A')}
  • Fecha de nacimiento: {driver.get('dob', 'N/A')}
  • Código: {driver.get('code', 'N/A')}
  • Número: {driver.get('number', 'N/A')}
📊 ESTADÍSTICAS DE CARRERA:
  • Carreras disputadas: {races}
  • Victorias: {wins}
  • Podios: {podiums}
  • Puntos totales: {points:.1f}
  • Posición promedio: {avg_pos:.2f if avg_pos > 0 else 'N/A'}
  • Tasa de podios: {(podiums/races*100):.1f}% if races > 0 else 0
🏆 RENDIMIENTO:
  • Victorias por carrera: {(wins/races*100):.2f}% if races > 0 else 0
  • Puntos por carrera: {(points/races):.2f if races > 0 else 0}
"""
            info_text.delete(1.0, tk.END)
            info_text.insert(tk.END, details)

        listbox.bind('<<ListboxSelect>>', show_driver_details)
        btn_frame = tk.Frame(search_window, bg="#1a1a1a")
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="🔍 Buscar", command=do_search, font=("Arial", 12, "bold"), bg="#e10600", fg="white", cursor="hand2", padx=35, pady=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cerrar", command=search_window.destroy, font=("Arial", 12), bg="#555555", fg="white", cursor="hand2", padx=35, pady=10).pack(side=tk.LEFT, padx=5)
        entry.bind("<Return>", lambda e: do_search())

    # ============================================================
    # PREDICCIONES ML
    # ============================================================
    def show_2025_predictions(self):
        if self.datos_2024.empty:
            messagebox.showwarning("Advertencia", "No hay predicciones disponibles")
            return
        top_10 = self.datos_2024.head(10)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
        colors = ['#FFD700', '#C0C0C0', '#CD7F32'] + ['#e10600']*7
        bars = ax1.barh(top_10['nombre_piloto'], top_10['prediccion_2025'], color=colors)
        ax1.set_xlabel('Posición Predicha 2025', fontsize=13)
        ax1.set_title('🔮 Top 10 Predicho - Campeonato F1 2025', fontsize=16, fontweight='bold')
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        ax1.grid(axis='x', alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax1.text(width - 0.3, bar.get_y() + bar.get_height()/2, f'{width:.1f}',
                    ha='right', va='center', fontweight='bold', fontsize=11, color='white')
        x = np.arange(len(top_10))
        width = 0.35
        ax2.bar(x - width/2, top_10['pos_ant'], width, label='Posición 2024', color='#00D2BE')
        ax2.bar(x + width/2, top_10['prediccion_2025'], width, label='Predicción 2025', color='#9333ea')
        ax2.set_xlabel('Piloto', fontsize=12)
        ax2.set_ylabel('Posición', fontsize=12)
        ax2.set_title('Comparación 2024 vs Predicción 2025', fontsize=15, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_10['nombre_piloto'], rotation=45, ha='right', fontsize=10)
        ax2.legend(fontsize=11)
        ax2.invert_yaxis()
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "🏆 PREDICCIONES CAMPEONATO F1 2025 - Random Forest")

    def show_rf_model_performance(self):
        if not self.rf_model:
            messagebox.showerror("Error", "Modelo no disponible")
            return
        fig = plt.figure(figsize=(18, 10))
        ax1 = plt.subplot(2, 2, 1)
        ax1.axis('off')
        metrics_text = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║   RANDOM FOREST MODEL PERFORMANCE                                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Algoritmo: Random Forest Regressor                                   ║
║  Número de árboles: 300                                               ║
║                                                                       ║
║  MÉTRICAS DE VALIDACIÓN:                                              ║
║  ────────────────────────────                                         ║
║  • Año de prueba: {int(self.model_metrics['test_year'])}              ║
║  • MAE: {self.model_metrics['mae']:.2f}                               ║
║  • RMSE: {self.model_metrics['rmse']:.2f}                             ║
║  • R² Score: {self.model_metrics['r2']:.3f}                           ║
║                                                                       ║
║  INTERPRETACIÓN:                                                      ║
║  El modelo predice posiciones con un                                  ║
║  error promedio de {self.model_metrics['mae']:.1f} posiciones.        ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
        ax1.text(0.05, 0.5, metrics_text, fontsize=12, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        if hasattr(self.rf_model, 'feature_importances_'):
            ax2 = plt.subplot(2, 2, 2)
            features = ['Puntos año anterior', 'Posición año anterior', 'Ranking equipo']
            importances = self.rf_model.feature_importances_
            ax2.barh(features, importances, color=['#e10600', '#00D2BE', '#9333ea'])
            ax2.set_xlabel('Importancia', fontsize=12)
            ax2.set_title('Importancia de Variables', fontsize=14, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
        ax3 = plt.subplot(2, 2, 3)
        top_5 = self.datos_2024.head(5)
        colors_podium = ['#FFD700', '#C0C0C0', '#CD7F32', '#e10600', '#e10600']
        ax3.barh(top_5['nombre_piloto'], top_5['prediccion_2025'], color=colors_podium)
        ax3.set_xlabel('Posición Predicha', fontsize=12)
        ax3.set_title('Top 5 Predicho 2025', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        ax3.invert_xaxis()
        ax3.grid(axis='x', alpha=0.3)
        ax4 = plt.subplot(2, 2, 4)
        ax4.hist(self.datos_2024['prediccion_2025'], bins=20, color='#9333ea', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Posición Predicha', fontsize=12)
        ax4.set_ylabel('Frecuencia', fontsize=12)
        ax4.set_title('Distribución de Predicciones 2025', fontsize=14, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, f"📊 Rendimiento Modelo Random Forest - R²: {self.model_metrics['r2']:.3f}")

    def predict_driver_2025(self):
        if self.datos_2024.empty:
            messagebox.showwarning("Advertencia", "No hay predicciones")
            return
        predict_window = tk.Toplevel(self.root)
        predict_window.title("🔮 Predicción Individual 2025")
        predict_window.geometry("900x650")
        predict_window.configure(bg="#1a1a1a")
        tk.Label(predict_window, text="🔮 Predicción de Posición 2025", font=("Arial", 18, "bold"), bg="#1a1a1a", fg="#9333ea").pack(pady=18)
        tk.Label(predict_window, text="Ingresa el nombre del piloto:", font=("Arial", 12), bg="#1a1a1a", fg="white").pack(pady=8)
        entry = tk.Entry(predict_window, font=("Arial", 13), width=50)
        entry.pack(pady=12)
        entry.focus()
        result_text = scrolledtext.ScrolledText(predict_window, font=("Courier", 11), bg="#2d2d2d", fg="white", height=25, wrap=tk.WORD)
        result_text.pack(fill=tk.BOTH, expand=True, padx=25, pady=12)

        def do_predict():
            search_term = entry.get().strip()
            if not search_term:
                messagebox.showwarning("Advertencia", "Ingresa un nombre")
                return
            lista_pilotos = self.datos_2024['nombre_piloto'].unique().tolist()
            exact_match = [p for p in lista_pilotos if search_term.lower() in p.lower()]
            if exact_match:
                matches = exact_match
            else:
                matches = difflib.get_close_matches(search_term, lista_pilotos, n=5, cutoff=0.4)
            if not matches:
                result_text.delete(1.0, tk.END)
                result_text.insert(tk.END, f"❌ No se encontró ningún piloto similar a '{search_term}'")
                return
            result_text.delete(1.0, tk.END)
            if len(matches) > 1:
                result_text.insert(tk.END, f"🔍 Se encontraron {len(matches)} pilotos:\n")
            for i, piloto_nombre in enumerate(matches, 1):
                fila = self.datos_2024[self.datos_2024['nombre_piloto'] == piloto_nombre].iloc[0]
                output = f"""{'='*70}
{i}. {piloto_nombre}
{'='*70}
🏎️  EQUIPO: {fila['constructor']}
📊 TEMPORADA 2024:
  • Posición final: {int(fila['pos_ant'])}°
  • Puntos obtenidos: {int(fila['pts_ant'])}
  • Ranking del equipo: {int(fila['rank_team_ant'])}°
🔮 PREDICCIÓN 2025:
  • Posición predicha: {fila['prediccion_2025']:.1f}°
  • Cambio esperado: {(fila['pos_ant'] - fila['prediccion_2025']):+.1f} posiciones
💡 ANÁLISIS:
"""
                cambio = fila['pos_ant'] - fila['prediccion_2025']
                if cambio > 2:
                    output += "  ✅ Se espera una mejora significativa\n"
                elif cambio > 0:
                    output += "  📈 Ligera mejora esperada\n"
                elif cambio < -2:
                    output += "  ⚠️ Se espera un retroceso significativo\n"
                elif cambio < 0:
                    output += "  📉 Ligero retroceso esperado\n"
                else:
                    output += "  ➡️ Rendimiento similar esperado\n"
                output += "\n"
                result_text.insert(tk.END, output)

        btn_frame = tk.Frame(predict_window, bg="#1a1a1a")
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="🔮 Predecir", command=do_predict, font=("Arial", 12, "bold"), bg="#9333ea", fg="white", cursor="hand2", padx=35, pady=10).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cerrar", command=predict_window.destroy, font=("Arial", 12), bg="#555555", fg="white", cursor="hand2", padx=35, pady=10).pack(side=tk.LEFT, padx=5)
        entry.bind("<Return>", lambda e: do_predict())

    # ============================================================
    # ANÁLISIS DE PILOTOS
    # ============================================================
    def show_top_winners(self):
        if self.results.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        wins = self.results[self.results['positionOrder'] == 1].copy()
        driver_wins = wins.groupby('driverId').size().reset_index(name='victories')
        driver_wins = driver_wins.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId')
        driver_wins['driver_name'] = driver_wins['forename'] + ' ' + driver_wins['surname']
        top_10 = driver_wins.nlargest(10, 'victories')
        fig, ax = plt.subplots(figsize=(16, 9))
        bars = ax.barh(top_10['driver_name'], top_10['victories'], color='#e10600')
        ax.set_xlabel('Victorias', fontsize=13)
        ax.set_title('🏆 Top 10 Pilotos con Más Victorias en F1', fontsize=18, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                   ha='left', va='center', fontweight='bold', fontsize=12)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "🏆 Top 10 Victorias")

    def show_top_podiums(self):
        if self.results.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        podiums = self.results[self.results['positionOrder'].isin([1, 2, 3])].copy()
        driver_podiums = podiums.groupby('driverId').size().reset_index(name='podiums')
        driver_podiums = driver_podiums.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId')
        driver_podiums['driver_name'] = driver_podiums['forename'] + ' ' + driver_podiums['surname']
        top_10 = driver_podiums.nlargest(10, 'podiums')
        fig, ax = plt.subplots(figsize=(16, 9))
        colors = ['#FFD700', '#C0C0C0', '#CD7F32'] * 4
        bars = ax.barh(top_10['driver_name'], top_10['podiums'], color=colors[:len(top_10)])
        ax.set_xlabel('Podios', fontsize=13)
        ax.set_title('🥇 Top 10 Pilotos con Más Podios', fontsize=18, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 5, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                   ha='left', va='center', fontweight='bold', fontsize=12)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "🥇 Top 10 Podios")

    def show_top_points(self):
        if self.results.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        driver_points = self.results.groupby('driverId')['points'].sum().reset_index()
        driver_points = driver_points.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId')
        driver_points['driver_name'] = driver_points['forename'] + ' ' + driver_points['surname']
        top_10 = driver_points.nlargest(10, 'points')
        fig, ax = plt.subplots(figsize=(16, 9))
        bars = ax.barh(top_10['driver_name'], top_10['points'], color='#00D2BE')
        ax.set_xlabel('Puntos Totales', fontsize=13)
        ax.set_title('📊 Top 10 Pilotos con Más Puntos', fontsize=18, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 50, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                   ha='left', va='center', fontweight='bold', fontsize=12)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "📊 Top 10 Puntos")

    # ============================================================
    # ANÁLISIS DE EQUIPOS
    # ============================================================
    def show_top_constructor_wins(self):
        if self.results.empty or self.constructors.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        wins = self.results[self.results['positionOrder'] == 1]
        constructor_wins = wins.groupby('constructorId').size().reset_index(name='victories')
        constructor_wins = constructor_wins.merge(self.constructors[['constructorId', 'name']], on='constructorId')
        top_10 = constructor_wins.nlargest(10, 'victories')
        fig, ax = plt.subplots(figsize=(16, 9))
        bars = ax.barh(top_10['name'], top_10['victories'], color='#1e3a8a')
        ax.set_xlabel('Victorias', fontsize=13)
        ax.set_title('🏆 Top 10 Equipos con Más Victorias', fontsize=18, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 3, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                   ha='left', va='center', fontweight='bold', fontsize=12)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "🏆 Top 10 Equipos - Victorias")

    def show_top_constructor_points(self):
        if self.results.empty or self.constructors.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        constructor_points = self.results.groupby('constructorId')['points'].sum().reset_index()
        constructor_points = constructor_points.merge(self.constructors[['constructorId', 'name']], on='constructorId')
        top_10 = constructor_points.nlargest(10, 'points')
        fig, ax = plt.subplots(figsize=(16, 9))
        bars = ax.barh(top_10['name'], top_10['points'], color='#16a34a')
        ax.set_xlabel('Puntos', fontsize=13)
        ax.set_title('📊 Top 10 Equipos con Más Puntos', fontsize=18, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 100, bar.get_y() + bar.get_height()/2, f'{int(width)}',
                   ha='left', va='center', fontweight='bold', fontsize=12)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "📊 Top 10 Equipos - Puntos")

    def show_constructor_evolution(self):
        if self.results.empty or self.constructors.empty or self.races.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        search_window = tk.Toplevel(self.root)
        search_window.title("Evolución de Equipo")
        search_window.geometry("550x250")
        search_window.configure(bg="#1a1a1a")
        tk.Label(search_window, text="Ingresa el nombre del equipo:", font=("Arial", 12), bg="#1a1a1a", fg="white").pack(pady=15)
        entry = tk.Entry(search_window, font=("Arial", 13), width=45)
        entry.pack(pady=10)
        entry.focus()
        def do_search():
            search_term = entry.get().strip()
            if not search_term:
                messagebox.showwarning("Advertencia", "Debes ingresar un nombre")
                return
            matches = self.constructors[
                self.constructors['name'].str.contains(search_term, case=False, na=False)
            ]
            if matches.empty:
                messagebox.showinfo("No encontrado", f"No se encontró ningún equipo con '{search_term}'")
                return
            constructor = matches.iloc[0]
            const_results = self.results[self.results['constructorId'] == constructor['constructorId']]
            const_results = const_results.merge(self.races[['raceId', 'year']], on='raceId', how='left')
            if const_results.empty:
                messagebox.showinfo("Sin datos", f"No hay datos históricos para {constructor['name']}")
                return

            yearly_stats = const_results.groupby('year').agg({
                'points': 'sum',
                'positionOrder': lambda x: (x == 1).sum()
            }).reset_index()
            yearly_stats.columns = ['year', 'points', 'wins']

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11))
            ax1.plot(yearly_stats['year'], yearly_stats['points'],
                    marker='o', linewidth=3, color='#e10600', markersize=8)
            ax1.set_xlabel('Año', fontsize=13)
            ax1.set_ylabel('Puntos', fontsize=13)
            ax1.set_title(f'Evolución de Puntos - {constructor["name"]}',
                         fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', which='major', labelsize=11)

            ax2.bar(yearly_stats['year'], yearly_stats['wins'],
                   color='#00D2BE', alpha=0.8, width=0.8)
            ax2.set_xlabel('Año', fontsize=13)
            ax2.set_ylabel('Victorias', fontsize=13)
            ax2.set_title(f'Victorias por Año - {constructor["name"]}',
                         fontsize=16, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.tick_params(axis='both', which='major', labelsize=11)

            plt.tight_layout()
            search_window.destroy()
            self.show_plot_in_main_area(fig, f"📈 Evolución de {constructor['name']}")
        tk.Button(search_window, text="Analizar", command=do_search, font=("Arial", 12, "bold"),
                  bg="#e10600", fg="white", cursor="hand2", padx=35, pady=10).pack(pady=15)
        entry.bind("<Return>", lambda e: do_search())

    # ============================================================
    # COMPARACIONES
    # ============================================================
    def compare_drivers(self):
        if self.results.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        compare_window = tk.Toplevel(self.root)
        compare_window.title("Comparar Pilotos")
        compare_window.geometry("550x350")
        compare_window.configure(bg="#1a1a1a")
        tk.Label(compare_window, text="Comparación de Pilotos", font=("Arial", 16, "bold"), bg="#1a1a1a", fg="white").pack(pady=15)
        tk.Label(compare_window, text="Primer piloto (nombre o apellido):", font=("Arial", 11), bg="#1a1a1a", fg="white").pack(pady=5)
        entry1 = tk.Entry(compare_window, font=("Arial", 12), width=40)
        entry1.pack(pady=8)
        tk.Label(compare_window, text="Segundo piloto (nombre o apellido):", font=("Arial", 11), bg="#1a1a1a", fg="white").pack(pady=5)
        entry2 = tk.Entry(compare_window, font=("Arial", 12), width=40)
        entry2.pack(pady=8)

        def do_compare():
            driver1_name = entry1.get().strip()
            driver2_name = entry2.get().strip()
            if not driver1_name or not driver2_name:
                messagebox.showwarning("Advertencia", "Debes ingresar ambos nombres")
                return
            driver1 = self.drivers[
                self.drivers['forename'].str.contains(driver1_name, case=False, na=False) |
                self.drivers['surname'].str.contains(driver1_name, case=False, na=False)
            ]
            driver2 = self.drivers[
                self.drivers['forename'].str.contains(driver2_name, case=False, na=False) |
                self.drivers['surname'].str.contains(driver2_name, case=False, na=False)
            ]
            if driver1.empty or driver2.empty:
                messagebox.showerror("Error", "No se encontraron uno o ambos pilotos")
                return
            d1 = driver1.iloc[0]
            d2 = driver2.iloc[0]
            d1_results = self.results[self.results['driverId'] == d1['driverId']]
            d2_results = self.results[self.results['driverId'] == d2['driverId']]
            d1_name = f"{d1.get('forename','')} {d1.get('surname','')}"
            d2_name = f"{d2.get('forename','')} {d2.get('surname','')}"

            stats = {
                'Carreras': [len(d1_results), len(d2_results)],
                'Victorias': [
                    len(d1_results[d1_results['positionOrder'] == 1]),
                    len(d2_results[d2_results['positionOrder'] == 1])
                ],
                'Podios': [
                    len(d1_results[d1_results['positionOrder'].isin([1, 2, 3])]),
                    len(d2_results[d2_results['positionOrder'].isin([1, 2, 3])])
                ],
                'Puntos': [
                    d1_results['points'].sum() if not d1_results.empty else 0,
                    d2_results['points'].sum() if not d2_results.empty else 0
                ]
            }

            fig, axes = plt.subplots(2, 2, figsize=(18, 11))
            colors = ['#e10600', '#00D2BE']
            metrics = ['Carreras', 'Victorias', 'Podios', 'Puntos']
            for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
                values = stats[metric]
                bars = ax.bar([d1_name, d2_name], values, color=colors, width=0.6)
                ax.set_ylabel(metric, fontsize=13)
                ax.set_title(metric, fontsize=15, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=11)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(values)*0.02,
                           f'{int(height)}', ha='center', va='bottom',
                           fontweight='bold', fontsize=12)

            plt.suptitle(f'Comparación: {d1_name} vs {d2_name}',
                        fontsize=18, fontweight='bold', y=0.995)
            plt.tight_layout()
            compare_window.destroy()
            self.show_plot_in_main_area(fig, f"⚔️ {d1_name} vs {d2_name}")

        tk.Button(compare_window, text="Comparar", command=do_compare, font=("Arial", 12, "bold"),
                  bg="#e10600", fg="white", cursor="hand2", padx=35, pady=10).pack(pady=18)

    # ============================================================
    # VISUALIZACIONES
    # ============================================================
    def show_heatmap(self):
        if self.results.empty or self.races.empty or self.drivers.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        wins = self.results[self.results['positionOrder'] == 1].copy()
        wins = wins.merge(self.races[['raceId', 'year']], on='raceId', how='left')
        wins = wins.merge(self.drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
        wins['driver_name'] = wins['forename'].fillna('') + ' ' + wins['surname'].fillna('')
        wins['decade'] = (wins['year'] // 10) * 10
        top_drivers = wins['driver_name'].value_counts().head(10).index
        decade_wins = wins[wins['driver_name'].isin(top_drivers)].groupby(
            ['driver_name', 'decade']
        ).size().reset_index(name='wins')
        pivot_table = decade_wins.pivot(index='driver_name', columns='decade', values='wins').fillna(0)

        fig, ax = plt.subplots(figsize=(18, 10))
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd',
                   cbar_kws={'label': 'Victorias'}, ax=ax,
                   annot_kws={'fontsize': 11})
        ax.set_title('Victorias por Década - Top 10 Pilotos',
                    fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel('Década', fontsize=13)
        ax.set_ylabel('Piloto', fontsize=13)
        ax.tick_params(axis='both', which='major', labelsize=11)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "🔥 Mapa de Calor - Victorias por Década")

    def show_distribution(self):
        if self.results.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        fig, ax = plt.subplots(figsize=(18, 9))
        position_counts = self.results['positionOrder'].value_counts().sort_index().head(20)
        bars = ax.bar(position_counts.index.astype(str), position_counts.values,
                     color='#9333ea', alpha=0.8, width=0.7)
        ax.set_xlabel('Posición Final', fontsize=13)
        ax.set_ylabel('Frecuencia', fontsize=13)
        ax.set_title('Distribución de Posiciones Finales en F1',
                    fontsize=18, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "📊 Distribución de Posiciones Finales")

    def show_top5_evolution(self):
        if self.results.empty or self.drivers.empty or self.races.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return
        driver_points = self.results.groupby('driverId')['points'].sum().reset_index()
        top5_drivers = driver_points.nlargest(5, 'points')
        fig, ax = plt.subplots(figsize=(18, 10))

        for _, row in top5_drivers.iterrows():
            driver_id = row['driverId']
            driver_info = self.drivers[self.drivers['driverId'] == driver_id]
            if driver_info.empty:
                continue
            driver_info = driver_info.iloc[0]
            driver_name = f"{driver_info.get('forename', '')} {driver_info.get('surname', '')}"
            driver_results = self.results[self.results['driverId'] == driver_id].copy()
            driver_results = driver_results.merge(self.races[['raceId', 'year']], on='raceId', how='left')
            if driver_results.empty:
                continue
            yearly_points = driver_results.groupby('year')['points'].sum().reset_index()
            ax.plot(yearly_points['year'], yearly_points['points'],
                    marker='o', linewidth=2.5, markersize=6, label=driver_name)

        ax.set_xlabel('Año', fontsize=13)
        ax.set_ylabel('Puntos por Temporada', fontsize=13)
        ax.set_title('Evolución de Puntos - Top 5 Pilotos Históricos',
                    fontsize=18, fontweight='bold')
        ax.legend(fontsize=12, title="Pilotos", title_fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=11)
        plt.tight_layout()
        self.show_plot_in_main_area(fig, "📈 Evolución Top 5 Pilotos")

    def show_general_stats(self):
        if self.drivers.empty:
            messagebox.showwarning("Datos", "Datos insuficientes")
            return

        nationality_counts = self.drivers['nationality'].value_counts()
        top_nat = nationality_counts.index[0] if not nationality_counts.empty else 'N/A'
        top_nat_count = nationality_counts.iloc[0] if not nationality_counts.empty else 0
        winners = self.results[self.results['positionOrder'] == 1]['driverId'].nunique() if not self.results.empty else 0
        podium_finishers = self.results[self.results['positionOrder'].isin([1,2,3])]['driverId'].nunique() if not self.results.empty else 0

        stats = f"""
{'=' * 80}
  ESTADÍSTICAS GENERALES - F1 ANALYZER PRO
{'=' * 80}
📊 DATASET:
  • Pilotos: {len(self.drivers)}
  • Carreras: {len(self.races)}
  • Equipos: {len(self.constructors)}
  • Vueltas registradas: {len(self.lap_times)}
  • Período: {self.races['year'].min() if not self.races.empty else 'N/A'} - {self.races['year'].max() if not self.races.empty else 'N/A'}
  • Nacionalidad más común: {top_nat} ({top_nat_count} pilotos)
  • Pilotos con al menos 1 victoria: {winners}
  • Pilotos con al menos 1 podio: {podium_finishers}
🤖 MODELO RANDOM FOREST:
  • Estado: {'✓ Activo' if self.rf_model else '✗ Inactivo'}
  • MAE: {self.model_metrics.get('mae', 0):.2f}
  • RMSE: {self.model_metrics.get('rmse', 0):.2f}
  • R² Score: {self.model_metrics.get('r2', 0):.3f}
  • Año de validación: {int(self.model_metrics.get('test_year', 0))}
🔮 PREDICCIONES 2025:
  • Pilotos con predicción: {len(self.datos_2024)}
  • Campeón predicho: {self.datos_2024.iloc[0]['nombre_piloto'] if not self.datos_2024.empty else 'N/A'}
{'=' * 80}
"""
        self.show_text_result(stats, "📈 Estadísticas Generales")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = F1AnalyzerGUI(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error Fatal", f"Error al iniciar:\n{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)