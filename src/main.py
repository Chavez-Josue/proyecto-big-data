import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual de las gráficas
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('crest')

def cargar_o_generar_datos(filepath: str) -> pd.DataFrame:
    """Carga el dataset CSV o lo genera sintéticamente si no existe."""
    if os.path.exists(filepath):
        print(f"[+] Cargando dataset desde: {filepath}")
        df = pd.read_csv(filepath)
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df

    print("[!] Dataset no encontrado. Generando datos sintéticos de prueba...")
    np.random.seed(42)
    n_registros = 1000

    fechas = pd.date_range(start='2025-01-01', end='2025-12-31', periods=n_registros)
    categorias = ['Electrónica', 'Ropa', 'Hogar', 'Juguetes', 'Libros']
    
    df = pd.DataFrame({
        'id_transaccion': range(1001, 1001 + n_registros),
        'fecha': fechas,
        'categoria': np.random.choice(categorias, size=n_registros, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'precio_unitario': np.round(np.random.uniform(10.0, 500.0, size=n_registros), 2),
        'cantidad': np.random.randint(1, 6, size=n_registros),
        'satisfaccion_cliente': np.random.choice([1, 2, 3, 4, 5, np.nan], size=n_registros, p=[0.05, 0.05, 0.15, 0.35, 0.35, 0.05])
    })
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[+] Dataset sintético guardado en: {filepath}")
    return df

def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta tareas esenciales de limpieza de datos."""
    print("\n--- INICIANDO LIMPIEZA DE DATOS ---")
    print(f"Registros iniciales: {len(df)}")
    
    # Imputación de valores faltantes en satisfacción con la mediana
    mediana_sat = df['satisfaccion_cliente'].median()
    df['satisfaccion_cliente'] = df['satisfaccion_cliente'].fillna(mediana_sat)
    
    # Cálculo de métrica derivada
    df['monto_total'] = df['precio_unitario'] * df['cantidad']
    
    print("Valores nulos limpios. Columna 'monto_total' calculada exitosamente.")
    return df

def analisis_descriptivo(df: pd.DataFrame):
    """Calcula y muestra en consola las métricas descriptivas principales."""
    print("\n==========================================")
    print("      RESUMEN DE ANALÍTICA DESCRIPTIVA    ")
    print("==========================================")
    
    total_ventas = df['monto_total'].sum()
    ticket_promedio = df['monto_total'].mean()
    total_transacciones = len(df)
    
    print(f"Total Ingresos Generados : ${total_ventas:,.2f}")
    print(f"Ticket Promedio de Compra: ${ticket_promedio:,.2f}")
    print(f"Total Transacciones      : {total_transacciones}")
    
    print("\n--- Ventas Totales por Categoría ---")
    ventas_cat = df.groupby('categoria')['monto_total'].agg(['sum', 'mean', 'count']).rename(
        columns={'sum': 'Total ($)', 'mean': 'Promedio ($)', 'count': 'Cant. Ventas'}
    )
    print(ventas_cat.sort_values(by='Total ($)', ascending=False))

def generar_visualizaciones(df: pd.DataFrame, output_dir: str):
    """Genera y guarda 3 gráficas clave para responder '¿Qué ocurrió?'."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Gráfica 1: Ventas totales por categoría
    plt.figure(figsize=(8, 5))
    ventas_cat = df.groupby('categoria')['monto_total'].sum().sort_values(ascending=False)
    ax = sns.barplot(x=ventas_cat.index, y=ventas_cat.values, color='#2b5c8f')
    plt.title('Ingresos Totales por Categoría de Producto', fontsize=12, fontweight='bold')
    plt.xlabel('Categoría')
    plt.ylabel('Monto Total ($)')
    for p in ax.patches:
        ax.annotate(f'${p.get_height():,.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    path_g1 = os.path.join(output_dir, 'ventas_por_categoria.png')
    plt.savefig(path_g1)
    plt.close()

    # Gráfica 2: Tendencia mensual de ingresos
    plt.figure(figsize=(10, 5))
    df_mes = df.set_index('fecha').resample('ME')['monto_total'].sum().reset_index()
    sns.lineplot(data=df_mes, x='fecha', y='monto_total', marker='o', color='#e74c3c', linewidth=2.5)
    plt.title('Tendencia Mensual de Ventas (2025)', fontsize=12, fontweight='bold')
    plt.xlabel('Mes')
    plt.ylabel('Ingresos ($)')
    plt.tight_layout()
    path_g2 = os.path.join(output_dir, 'tendencia_mensual.png')
    plt.savefig(path_g2)
    plt.close()

    # Gráfica 3: Distribución de satisfacción según precio unitario
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='satisfaccion_cliente', y='precio_unitario', palette='Blues')
    plt.title('Distribución de Precio Unitario por Nivel de Satisfacción', fontsize=12, fontweight='bold')
    plt.xlabel('Nivel de Satisfacción (1 a 5)')
    plt.ylabel('Precio Unitario ($)')
    plt.tight_layout()
    path_g3 = os.path.join(output_dir, 'precio_vs_satisfaccion.png')
    plt.savefig(path_g3)
    plt.close()

    print(f"\n[+] Visualizaciones guardadas con éxito en: {output_dir}")

def main():
    data_path = os.path.join('data', 'ventas_ecommerce.csv')
    docs_dir = os.path.join('docs', 'evidencias')
    
    df = cargar_o_generar_datos(data_path)
    df = limpiar_datos(df)
    analisis_descriptivo(df)
    generar_visualizaciones(df, docs_dir)

if __name__ == '__main__':
    main()