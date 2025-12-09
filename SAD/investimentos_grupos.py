import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("investimentos2.csv", sep=';', encoding='latin1')


df['perfil_recomendado'] = None


mask_reserva = (df['risco'] <= 1) & (df['liquidez_dias'] == 0)
df.loc[mask_reserva, 'perfil_recomendado'] = "Reserva/Proteção Total"

mask_conservador = (df['risco'] == 2)
df.loc[mask_conservador, 'perfil_recomendado'] = "Conservador Estruturado"


mask_turbo = (df['risco'] >= 4) & (df['liquidez_dias'] <= 15)
indices = df[mask_turbo].index[:6]
df.loc[indices, 'perfil_recomendado'] = "Turbo Arrojado"


resto_df = df[df['perfil_recomendado'].isna()].copy()
features = ['risco','liquidez_dias','horizonte_min_meses','retorno_esperado_anual','rendimento_mensal','volatilidade']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(resto_df[features])

kmeans = KMeans(n_clusters=2, random_state=42)
resto_df['cluster'] = kmeans.fit_predict(X_scaled)


cluster_means = resto_df.groupby('cluster')[['risco','horizonte_min_meses','liquidez_dias','volatilidade']].mean()

perfil_stats = {
    "Arrojado Estratégico": {"risco":4, "horizonte_min_meses":12, "liquidez_dias":30, "volatilidade":6},
    "Moderado Tático": {"risco":3, "horizonte_min_meses":36, "liquidez_dias":60, "volatilidade":3},
}

def perfil_mais_proximo(row):
    min_dist = float('inf')
    perfil_escolhido = None
    for perfil, stats in perfil_stats.items():
        dist = np.sqrt(
            2*(row['risco']-stats['risco'])**2 +
            1*((row['horizonte_min_meses']-stats['horizonte_min_meses'])/12)**2 +
            1*((row['liquidez_dias']-stats['liquidez_dias'])/30)**2 +
            1*((row['volatilidade']-stats['volatilidade'])/2)**2
        )
        if dist < min_dist:
            min_dist = dist
            perfil_escolhido = perfil
    return perfil_escolhido

cluster_to_perfil = {}
for cluster_id in cluster_means.index:
    cluster_to_perfil[cluster_id] = perfil_mais_proximo(cluster_means.loc[cluster_id])

resto_df['perfil_recomendado'] = resto_df['cluster'].map(cluster_to_perfil)


def limitar_perfis(df, perfil, limite):
    indices = df[df['perfil_recomendado']==perfil].index
    if len(indices) > limite:
        to_change = indices[limite:]

        df.loc[to_change, 'perfil_recomendado'] = 'Moderado Tático' if perfil=='Arrojado Estratégico' else 'Arrojado Estratégico'
    return df

resto_df = limitar_perfis(resto_df, 'Moderado Tático', 25)
resto_df = limitar_perfis(resto_df, 'Arrojado Estratégico', 6)
resto_df = limitar_perfis(resto_df, 'Turbo Arrojado', 6)


final_df = pd.concat([df[df['perfil_recomendado'].notna()], resto_df]).reset_index(drop=True)

X_all = scaler.fit_transform(final_df[features])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_all)
final_df['PCA1'] = pca_result[:,0]
final_df['PCA2'] = pca_result[:,1]


plt.figure(figsize=(10,7))
for perfil in final_df['perfil_recomendado'].unique():
    subset = final_df[final_df['perfil_recomendado']==perfil]
    plt.scatter(subset['PCA1'], subset['PCA2'], s=100, label=f"{perfil} ({len(subset)})")
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('Visualização de investimentos por perfil (PCA)')
plt.legend()
plt.grid(True)
plt.show()


for perfil in final_df['perfil_recomendado'].unique():
    subset = final_df[final_df['perfil_recomendado']==perfil]
    print(f"\nPerfil: {perfil} ({len(subset)} investimentos)")
    print(subset[['nome', 'classe', 'risco_investimento', 'liquidez_dias', 'horizonte_min_meses', 'volatilidade']])

final_df.to_csv("investimentos_clusterizados_pca.csv", sep=';', index=False, encoding='utf-8')


