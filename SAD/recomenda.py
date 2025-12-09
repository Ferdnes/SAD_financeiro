
import pandas as pd
import joblib
import numpy as np

rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

df_inv = pd.read_csv("investimentos_clusterizados_pca.csv", sep=';', encoding='utf-8')


def calcular_score(row):
    score = 0
    justificativa = []

 
    if row["idade"] < -0.5:
        score += 2
        justificativa.append("Investidor jovem, geralmente mais aberto a assumir riscos.")
    elif row["idade"] < 0.5:
        score += 1
        justificativa.append("Idade intermediária, alguma disposição para risco.")
    else:
        justificativa.append("Idade mais alta, tendência a maior cautela.")

    educacao = str(row.get('educacao_financeiro', '')).lower()
    if 'frequentemente' in educacao:
        score += 2
        justificativa.append("Investidor com bom conhecimento financeiro, entende riscos e oportunidades.")
    elif 'raramente' in educacao:
        score += 1
        justificativa.append("Conhecimento financeiro moderado, precisa de orientação em estratégias mais complexas.")
    else:
        justificativa.append("Pouco conhecimento financeiro, tende a preferir segurança.")

    risco = str(row.get('risco', '')).lower()
    risco_map = {
        'muito baixo': (-2, "Muito avesso a risco, prefere estabilidade absoluta."),
        'baixo': (-1, "Baixa tolerância a risco, prioriza segurança sobre retorno."),
        'moderado': (1, "Tolerância moderada a risco, busca equilíbrio entre retorno e segurança."),
        'alto': (3, "Alto risco: confortável com volatilidade e possibilidade de perdas temporárias."),
        'muito alto': (4, "Muito alto risco: disposto a grandes variações em busca de altos retornos.")
    }
    for chave, (peso, frase) in risco_map.items():
        if chave in risco:
            score += peso
            justificativa.append(frase)
            break
    else:
        justificativa.append("Perfil de risco não especificado, assume postura conservadora por padrão.")

    aceita_ir = str(row.get('aceita_ir', '')).lower()
    if 'sim, se o retorno for superior' in aceita_ir:
        score += 1
        justificativa.append("Aceita estratégias mais arriscadas se houver potencial de maior retorno.")


    dividas = str(row.get('dividas', '0'))
    if dividas == '1':
        score -= 2
        justificativa.append("Possui dívidas, reduz a capacidade de assumir riscos adicionais.")

    horizonte = str(row.get('horizonte', '')).lower()
    if 'longo prazo' in horizonte:
        score += 2
        justificativa.append("Horizonte de investimento longo, pode assumir riscos planejados.")
    elif 'médio prazo' in horizonte:
        score += 1
        justificativa.append("Horizonte médio, risco moderado é adequado.")
    elif 'curto prazo' in horizonte:
        justificativa.append("Investimentos de curto prazo tendem a ser mais sensíveis à volatilidade.")

  
    if ('alto' in risco or 'muito alto' in risco) and ('curto prazo' in horizonte):
        score += 2
        justificativa.append("Alto risco + horizonte curto: pronto para ganhos rápidos, mas maior volatilidade.")

    return score, justificativa


def recomendar_investimentos(novo_investidor: dict):

    idade_raw = np.array([[novo_investidor.get("idade", 30)]])
    idade_scaled = scaler.transform(idade_raw)[0][0]
    novo_investidor["idade"] = idade_scaled


    score, justificativa = calcular_score(novo_investidor)

    if score <= 3:
        perfil = "Conservador"
    elif score <= 6:
        perfil = "Moderado"
    elif score <= 9:
        perfil = "Arrojado"
    else:
        perfil = "Muito Arrojado"

    
    if perfil == "Conservador":
        top_df = df_inv[df_inv["perfil_recomendado"].str.contains("Conservador", na=False)].sample(3)
    elif perfil == "Moderado":
        top_df = df_inv[df_inv["perfil_recomendado"].str.contains("Moderado", na=False)].sample(3)
    elif perfil == "Arrojado":
        top_df = df_inv[df_inv["perfil_recomendado"].str.contains("Arrojado", na=False)].sample(3)
    else:  
        top_df = df_inv[df_inv["perfil_recomendado"].str.contains("Turbo|Muito Arrojado", na=False)].sample(3)

    top_df = top_df.fillna("").copy()
    for col in top_df.columns:
        if top_df[col].dtype.kind in "f":  
            top_df[col] = top_df[col].astype(float)
        elif top_df[col].dtype.kind in "i":  
            top_df[col] = top_df[col].astype(int)
        else:
            top_df[col] = top_df[col].astype(str)

    for idx, row in top_df.iterrows():
        banco = row.get("instituicao_bancaria", "Banco não informado")
        if perfil in ["Conservador", "Moderado"]:
            top_df.at[idx, "justificativa"] = f"{banco}: compatível com seu perfil {perfil}, prioriza segurança e estabilidade."
        elif perfil == "Arrojado":
            top_df.at[idx, "justificativa"] = f"{banco}: compatível com seu perfil {perfil}, busca maior retorno com risco moderado."
        else:
            top_df.at[idx, "justificativa"] = f"{banco}: compatível com seu perfil {perfil}, voltado a alto retorno e maior volatilidade."

    return perfil, justificativa, top_df
