
import pandas as pd
import joblib


rf = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

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
    if 'alto' in risco or 'muito alto' in risco:
        score += 3
        justificativa.append("Perfil de risco elevado, confortável com grandes variações.")
    elif 'moderado' in risco:
        score += 1
        justificativa.append("Perfil de risco moderado, busca equilíbrio entre retorno e segurança.")
    else:
        score -= 1
        justificativa.append("Perfil conservador, evita perdas e prefere segurança.")


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


    if ('alto' in risco) and ('curto prazo' in horizonte):
        score += 2
        justificativa.append("Alto risco + horizonte curto: pronto para ganhos rápidos, mas maior volatilidade.")

    return score, justificativa

def avaliar_investidor_hibrido(novo_investidor: dict, score_arrojado_min=8):
    df = pd.DataFrame([novo_investidor])


    categoricas = [
        'objetivo', 'reserva_de_emergencia', 'horizonte', 'liquidez',
        'educacao_financeiro', 'risco', 'aceita_ir'
    ]
    for col in categoricas:
        df[col] = df[col].astype(str).str.lower().str.strip().astype('category')

    df['dividas'] = df['dividas'].map(lambda x: 0 if 'não' in str(x).lower() else 1)
    df['idade'] = scaler.transform(df[['idade']])

    todas_categoricas = categoricas + ['dividas']
    df_encoded = pd.get_dummies(df, columns=todas_categoricas, drop_first=True)


    for col in rf.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[rf.feature_names_in_]


    perfil_rf = rf.predict(df_encoded)[0]
    probs_rf = dict(zip(rf.classes_, rf.predict_proba(df_encoded)[0]))


    score_manual, justificativa = calcular_score(novo_investidor)


    if score_manual >= score_arrojado_min:
        perfil_final = "Arrojado"
    else:
        perfil_final = perfil_rf

    return perfil_final, probs_rf, score_manual, justificativa
