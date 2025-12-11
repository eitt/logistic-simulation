# ==========================================
# App de Simulación Didáctica
# Autor: Leonardo H. Talero-Sarmiento
# ==========================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import time

# Configuración de página
st.set_page_config(
    page_title="Simulación y Optimización - L. Talero",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Funciones Auxiliares y Estilos
# ==========================================
def footer():
    st.markdown("---")
    st.markdown("Desarrollado con fines didácticos por **Leonardo H. Talero-Sarmiento** | [Ver perfil](https://apolo.unab.edu.co/en/persons/leonardo-talero)")

def local_css(file_name):
    # Placeholder si quisieras CSS personalizado
    pass

# ==========================================
# Módulo 1: Estimación de Pi (Montecarlo)
# ==========================================
def page_montecarlo_pi():
    st.title("🎲 Método de Montecarlo: Estimando $\pi$")
    st.markdown("""
    El método de Montecarlo utiliza el muestreo aleatorio para resolver problemas deterministas. 
    En este ejemplo, estimaremos el valor de $\pi$ "lanzando dardos" aleatorios a un cuadrado.
    
    **La Lógica:**
    1. Tenemos un cuadrado de lado $2$ (área $4$) y un círculo inscrito de radio $1$ (área $\pi$).
    2. La relación de áreas es: $\frac{\text{Área Círculo}}{\text{Área Cuadrado}} = \frac{\pi}{4}$.
    3. Por tanto, $\pi \approx 4 \times \frac{\text{Puntos dentro del círculo}}{\text{Total de puntos}}$.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configuración")
        n_points = st.slider("Número de Puntos (Simulaciones)", 100, 10000, 1000, 100)
        seed = st.number_input("Semilla Aleatoria (Seed)", value=42)
        run_btn = st.button("Ejecutar Simulación")

    if run_btn:
        np.random.seed(seed)
        # Generar puntos x, y entre -1 y 1
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        
        # Calcular distancia al origen
        dist = np.sqrt(x**2 + y**2)
        inside = dist <= 1
        
        pi_estimate = 4 * np.sum(inside) / n_points
        error = abs(np.pi - pi_estimate) / np.pi * 100

        with col2:
            st.metric(label="Valor estimado de Pi", value=f"{pi_estimate:.5f}", delta=f"Error: {error:.4f}%")
            
            # Gráfica
            fig = go.Figure()
            
            # Círculo guía
            fig.add_shape(type="circle", xref="x", yref="y", x0=-1, y0=-1, x1=1, y1=1, line_color="Black")
            
            # Puntos
            fig.add_trace(go.Scatter(
                x=x[inside], y=y[inside], mode='markers', name='Dentro',
                marker=dict(color='blue', size=5, opacity=0.6)
            ))
            fig.add_trace(go.Scatter(
                x=x[~inside], y=y[~inside], mode='markers', name='Fuera',
                marker=dict(color='red', size=5, opacity=0.6)
            ))
            
            fig.update_layout(
                title="Visualización de Disparos Aleatorios",
                xaxis=dict(range=[-1.1, 1.1], scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[-1.1, 1.1]),
                width=600, height=600,
                template="plotly_white"
            )
            st.plotly_chart(fig)

    st.info("💡 **Nota Didáctica:** Observa que a medida que aumentas el número de puntos (Ley de los Grandes Números), la estimación converge hacia el valor real de 3.14159...")

# ==========================================
# Módulo 2: Optimización (Gradiente vs Genético)
# ==========================================
def page_optimization():
    st.title("🏔️ Optimización: Descenso del Gradiente vs. Algorimo Genético")
    st.markdown("""
    Aquí comparamos dos enfoques para encontrar el mínimo de una función matemática:
    1.  **Descenso del Gradiente:** Como una pelota rodando cuesta abajo. Funciona genial en funciones convexas (forma de tazón).
    2.  **Algoritmo Genético:** Imita la evolución natural. Funciona mejor en funciones no convexas con múltiples "trampas" (óptimos locales).
    """)

    function_type = st.selectbox("Selecciona el tipo de función", ["Convexa (Cuadrática)", "No Convexa (Multimodal)"])

    # Definición de funciones
    x_range = np.linspace(-10, 10, 400)
    
    if function_type == "Convexa (Cuadrática)":
        # f(x) = x^2
        y = x_range**2
        func = lambda x: x**2
        grad = lambda x: 2*x
        st.latex(r"f(x) = x^2 \quad \text{(Solo un mínimo global en } x=0 \text{)}")
    else:
        # Función compleja: f(x) = x^2 + 10*sin(x) (aproximadamente) pero escalada para visualización
        # Usaremos una función tipo Rastrigin 1D simplificada
        func = lambda x: 0.1*x**2 + 2*np.sin(x)
        grad = lambda x: 0.2*x + 2*np.cos(x)
        y = func(x_range)
        st.latex(r"f(x) = 0.1x^2 + 2\sin(x) \quad \text{(Múltiples mínimos locales)}")

    col1, col2 = st.columns(2)

    # --- Descenso del Gradiente ---
    with col1:
        st.subheader("1. Descenso del Gradiente")
        lr = st.slider("Tasa de Aprendizaje (Learning Rate)", 0.01, 1.1, 0.1, 0.01)
        start_x = st.slider("Punto de inicio (x)", -9.0, 9.0, 8.0)
        iterations_gd = st.slider("Iteraciones GD", 1, 50, 20)

        # Algoritmo
        path_x = [start_x]
        path_y = [func(start_x)]
        curr_x = start_x
        for _ in range(iterations_gd):
            curr_x = curr_x - lr * grad(curr_x)
            path_x.append(curr_x)
            path_y.append(func(curr_x))

        # Plot
        fig_gd = go.Figure()
        fig_gd.add_trace(go.Scatter(x=x_range, y=y, mode='lines', name='Función', line=dict(color='gray')))
        fig_gd.add_trace(go.Scatter(x=path_x, y=path_y, mode='markers+lines', name='Trayectoria', 
                                    marker=dict(color='red', size=8), line=dict(dash='dot')))
        fig_gd.add_trace(go.Scatter(x=[path_x[-1]], y=[path_y[-1]], mode='markers', name='Final', 
                                    marker=dict(color='green', size=12, symbol='star')))
        
        fig_gd.update_layout(title="Trayectoria del Gradiente", xaxis_title="x", yaxis_title="f(x)")
        st.plotly_chart(fig_gd, use_container_width=True)
        
        if function_type == "No Convexa (Multimodal)":
            st.warning("⚠️ Nota como el Gradiente puede quedarse atascado en un hueco local si no inicia cerca del óptimo global.")

    # --- Algoritmo Genético ---
    with col2:
        st.subheader("2. Algoritmo Genético (AG)")
        pop_size = st.slider("Tamaño Población", 5, 50, 20)
        generations = st.slider("Generaciones", 1, 50, 10)
        mutation_rate = st.slider("Tasa de Mutación", 0.0, 1.0, 0.1)

        # Algoritmo Genético Simplificado
        np.random.seed(42)
        population = np.random.uniform(-10, 10, pop_size)
        
        history_pop = [population.copy()]
        
        for _ in range(generations):
            # Evaluar aptitud (fitness) - Queremos minimizar, así que fitness negativo o inverso
            fitness = -func(population) # Maximizamos el negativo para minimizar
            
            # Selección (Torneo simple)
            new_pop = []
            for _ in range(pop_size):
                i, j = np.random.randint(0, pop_size, 2)
                if fitness[i] > fitness[j]:
                    winner = population[i]
                else:
                    winner = population[j]
                new_pop.append(winner)
            
            new_pop = np.array(new_pop)
            
            # Cruce (Promedio simple para valores reales - Crossover)
            offspring = []
            for i in range(0, pop_size, 2):
                p1 = new_pop[i]
                p2 = new_pop[(i+1)%pop_size]
                c1 = (p1 + p2)/2
                c2 = (p1 + p2)/2 # Simplificación
                offspring.extend([c1, c2])
            
            offspring = np.array(offspring)

            # Mutación
            mask = np.random.rand(pop_size) < mutation_rate
            noise = np.random.normal(0, 2, pop_size) # Ruido gaussiano
            offspring[mask] += noise[mask]
            
            # Clip para mantener en rango
            population = np.clip(offspring, -10, 10)
            history_pop.append(population.copy())

        # Plot Final Generation
        final_pop_x = history_pop[-1]
        final_pop_y = func(final_pop_x)

        fig_ga = go.Figure()
        fig_ga.add_trace(go.Scatter(x=x_range, y=y, mode='lines', name='Función', line=dict(color='gray')))
        fig_ga.add_trace(go.Scatter(x=final_pop_x, y=final_pop_y, mode='markers', name='Población Final', 
                                    marker=dict(color='purple', size=8, opacity=0.7)))
        
        # Highlight best
        best_idx = np.argmin(final_pop_y)
        fig_ga.add_trace(go.Scatter(x=[final_pop_x[best_idx]], y=[final_pop_y[best_idx]], mode='markers', name='Mejor Individuo', 
                                    marker=dict(color='gold', size=14, symbol='diamond', line=dict(width=2, color='black'))))

        fig_ga.update_layout(title=f"AG - Generación {generations}", xaxis_title="x", yaxis_title="f(x)")
        st.plotly_chart(fig_ga, use_container_width=True)
        
        if function_type == "No Convexa (Multimodal)":
            st.success("✅ El AG explora múltiples áreas y tiene mayor probabilidad de 'saltar' fuera de óptimos locales.")

# ==========================================
# Módulo 3: Simulación de Procesos (CLT)
# ==========================================
def page_process_simulation():
    st.title("🏭 Simulación de Procesos y Teorema del Límite Central")
    st.markdown("""
    Imagina un proceso industrial con **5 actividades secuenciales**.
    Cada actividad tiene una duración variable (aleatoria).
    
    $$ T_{total} = t_1 + t_2 + t_3 + t_4 + t_5 $$
    
    Aunque cada actividad individual tenga una distribución extraña (Uniforme, Triangular), la suma total de los tiempos tenderá a comportarse como una **Distribución Normal**. ¡Vamos a probarlo simulando miles de escenarios!
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Configurar Actividades")
        n_scenarios = st.number_input("Número de Escenarios a simular", 100, 50000, 5000)
        
        dist_types = ["Uniforme (Min, Max)", "Triangular (Min, Moda, Max)", "Exponencial (Scale)"]
        
        activities = []
        for i in range(5):
            st.markdown(f"**Actividad {i+1}**")
            dtype = st.selectbox(f"Distribución A{i+1}", dist_types, key=f"d{i}")
            
            if "Uniforme" in dtype:
                mn = st.number_input(f"Min A{i+1}", 1, 50, 5 + i*2, key=f"mn{i}")
                mx = st.number_input(f"Max A{i+1}", mn+1, 100, 15 + i*2, key=f"mx{i}")
                activities.append({"type": "uniform", "params": (mn, mx)})
            elif "Triangular" in dtype:
                mn = st.number_input(f"Min A{i+1}", 1, 50, 5 + i*2, key=f"tmn{i}")
                md = st.number_input(f"Moda A{i+1}", mn, mn+20, mn+5, key=f"tmd{i}")
                mx = st.number_input(f"Max A{i+1}", md, 100, md+10, key=f"tmx{i}")
                activities.append({"type": "triangular", "params": (mn, md, mx)})
            else:
                scl = st.number_input(f"Escala A{i+1}", 1, 50, 10, key=f"exp{i}")
                activities.append({"type": "exponential", "params": (scl,)})
            st.markdown("---")

    with col2:
        if st.button("🚀 Simular Proceso", type="primary"):
            # Lógica de Simulación
            scenario_times = np.zeros(n_scenarios)
            
            # Simular actividad por actividad y sumar
            # Vectorizamos para velocidad
            for act in activities:
                if act["type"] == "uniform":
                    t = np.random.uniform(act["params"][0], act["params"][1], n_scenarios)
                elif act["type"] == "triangular":
                    t = np.random.triangular(act["params"][0], act["params"][1], act["params"][2], n_scenarios)
                elif act["type"] == "exponential":
                    t = np.random.exponential(act["params"][0], n_scenarios)
                
                scenario_times += t
            
            # Análisis de Resultados
            mu = np.mean(scenario_times)
            sigma = np.std(scenario_times)
            
            st.success(f"**Resultados de {n_scenarios} escenarios:**")
            c_a, c_b, c_c = st.columns(3)
            c_a.metric("Tiempo Promedio (Ciclo)", f"{mu:.2f} min")
            c_b.metric("Desviación Estándar", f"{sigma:.2f} min")
            c_c.metric("Min / Max Tiempo", f"{np.min(scenario_times):.1f} / {np.max(scenario_times):.1f}")

            # Histograma y Curva Normal Ajustada
            fig = px.histogram(scenario_times, nbins=50, title="Distribución del Tiempo Total de Ciclo",
                               labels={'value': 'Tiempo Total', 'count': 'Frecuencia'},
                               opacity=0.7, color_discrete_sequence=['teal'])
            
            # Añadir curva normal teórica
            x_axis = np.linspace(np.min(scenario_times), np.max(scenario_times), 1000)
            pdf = stats.norm.pdf(x_axis, mu, sigma)
            # Escalar pdf para que coincida con el histograma (aprox)
            # Nota: Esto es visual. Para precisión exacta usar histnorm='probability density'
            
            fig.add_trace(go.Scatter(x=x_axis, y=pdf * n_scenarios * (np.max(scenario_times)-np.min(scenario_times))/50, 
                                     mode='lines', name='Ajuste Normal (Teórico)',
                                     line=dict(color='firebrick', width=3)))

            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Interpretación Teórica
            Observa el histograma. Aunque hayas seleccionado distribuciones planas (Uniforme) o sesgadas (Triangular/Exponencial) para las actividades individuales, la suma total tiene esa forma de campana característica.
            
            Esto es el **Teorema del Límite Central (CLT)** en acción:
            $$ \sum_{i=1}^{n} X_i \xrightarrow{d} \mathcal{N}(\mu, \sigma^2) $$
            
            Esto nos permite aproximar probabilidades de cumplimiento (Service Levels) usando la normal estándar.
            """)
            
            # Cálculo de Probabilidad (Ejemplo)
            target = st.slider("¿Cuál es el tiempo meta de entrega?", float(np.min(scenario_times)), float(np.max(scenario_times)), float(mu))
            prob_cumplimiento = (scenario_times <= target).mean() * 100
            st.info(f"Probabilidad de terminar el proceso en **{target:.2f} minutos** o menos: **{prob_cumplimiento:.2f}%**")

# ==========================================
# Main Navigation
# ==========================================
def main():
    st.sidebar.title("Navegación")
    options = ["Inicio", "Estimación de Pi (Montecarlo)", "Optimización (Gradiente vs Genético)", "Simulación de Procesos (CLT)"]
    choice = st.sidebar.radio("Ir a:", options)

    if choice == "Inicio":
        st.title("Laboratorio de Simulación Computacional")
        st.markdown("""
        Bienvenido a esta aplicación interactiva diseñada para la enseñanza de **Simulación y Modelado**.
        
        Selecciona un módulo en el panel izquierdo para comenzar:
        
        1.  **Estimación de Pi:** Aprende los fundamentos del método Montecarlo.
        2.  **Optimización:** Compara heurísticas para funciones convexas y no convexas.
        3.  **Simulación de Procesos:** Entiende cómo la variabilidad se acumula en un sistema productivo y el Teorema del Límite Central.
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/8/84/Pi_30K.gif", caption="Visualización conceptual de Montecarlo")
    
    elif choice == "Estimación de Pi (Montecarlo)":
        page_montecarlo_pi()
    
    elif choice == "Optimización (Gradiente vs Genético)":
        page_optimization()
        
    elif choice == "Simulación de Procesos (CLT)":
        page_process_simulation()
    
    footer()

if __name__ == "__main__":
    main()