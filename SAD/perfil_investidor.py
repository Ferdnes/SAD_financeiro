import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib


investidores = pd.read_csv("investidores.csv", encoding='latin1', sep=';')


investidores.columns = investidores.columns.str.strip().str.lower().str.replace(' ', '_')
print("Colunas do DataFrame:", investidores.columns.tolist())


categoricas = [
    'objetivo', 'reserva_de_emergencia', 'horizonte', 'liquidez',
    'educacao_financeiro', 'risco', 'aceita_ir'
]
for col in categoricas:
    investidores[col] = investidores[col].astype(str).str.lower().str.strip().astype('category')

investidores['dividas'] = investidores['dividas'].map(lambda x: 0 if 'não' in str(x).lower() else 1)

investidores['idade'] = pd.to_numeric(investidores['idade'], errors='coerce')


for col in investidores.columns:
    if investidores[col].isnull().sum() > 0:
        if investidores[col].dtype.name == 'category':
            investidores[col] = investidores[col].fillna(investidores[col].mode()[0])
        else:
            investidores[col] = investidores[col].fillna(investidores[col].median())


scaler = StandardScaler()
investidores['idade'] = scaler.fit_transform(investidores[['idade']])


categorias_fixas = {
    'objetivo': ['construir reserva de emergência', 'comprar um bem (carro, casa, viagem)',
                 'acumular patrimônio no longo prazo', 'aposentadoria', 'gerar renda mensal'],
    'reserva_de_emergencia': ['não possuo', 'sim, menos de 6 meses de despesas', 'sim, mais de 6 meses'],
    'horizonte': ['curto prazo (até 1 ano)', 'médio prazo (1 a 5 anos)', 'longo prazo (acima de 5 anos)'],
    'liquidez': ['resgate diário', 'resgate em 30 dias', 'resgate em alguns meses',
                 'quero resgatar daqui a 1 ano', 'posso deixar por anos sem mexer'],
    'educacao_financeiro': ['nunca', 'raramente', 'frequentemente'],
    'risco': ['muito baixo', 'baixo', 'moderado', 'alto', 'muito alto'],
    'aceita_ir': ['não', 'sim, se o retorno for superior', 'não tenho preferência']
}
for col, cat_list in categorias_fixas.items():
    investidores[col] = pd.Categorical(investidores[col], categories=cat_list)


def risco_alto_horizonte_curto(row):
    risco = str(row['risco']).lower() if pd.notnull(row['risco']) else ""
    horizonte = str(row['horizonte']).lower() if pd.notnull(row['horizonte']) else ""
    if ('alto' in risco) and ('curto prazo' in horizonte):
        return 1
    else:
        return 0

investidores['alto_risco_curto_horizonte'] = investidores.apply(risco_alto_horizonte_curto, axis=1)


todas_categoricas = categoricas + ['dividas']
investidores_encoded = pd.get_dummies(investidores, columns=todas_categoricas, drop_first=True)


def tem_palavra(row, coluna_base, palavra):
    for col in row.index:
        if coluna_base in col and palavra in col and row[col] == 1:
            return True
    return False


def classificar_perfil_com_justificativa(row):
    score = 0
    justificativa = []

    if row["idade"] < -0.5:
        score += 2
        justificativa.append("Idade baixa: mais disposição para risco")
    elif row["idade"] < 0.5:
        score += 1
        justificativa.append("Idade média: alguma disposição para risco")


    if tem_palavra(row, "educacao_financeiro", "frequentemente"):
        score += 2
        justificativa.append("Alta educação financeira: entende riscos")
    elif tem_palavra(row, "educacao_financeiro", "raramente"):
        score += 1
        justificativa.append("Educação financeira moderada")

    if tem_palavra(row, "risco", "muito alto") or tem_palavra(row, "risco", "alto"):
        score += 3
        justificativa.append("Perfil de risco alto: aceita grandes variações")
    elif tem_palavra(row, "risco", "moderado"):
        score += 1
        justificativa.append("Perfil de risco moderado")
    else:
        score -= 1
        justificativa.append("Perfil de risco baixo: evita perdas")

    if tem_palavra(row, "aceita_ir", "sim, se o retorno for superior"):
        score += 1
        justificativa.append("Aceita estratégias mais arriscadas (IR)")


    if tem_palavra(row, "dividas", "1"):
        score -= 2
        justificativa.append("Possui dívidas: menor capacidade para arriscar")

    if tem_palavra(row, "horizonte", "longo prazo (acima de 5 anos)"):
        score += 2
        justificativa.append("Horizonte longo: pode assumir riscos planejados")
    elif tem_palavra(row, "horizonte", "médio prazo (1 a 5 anos)"):
        score += 1
        justificativa.append("Horizonte médio")

    
    if score <= 2:
        perfil = "Conservador"
    elif score <= 5:
        perfil = "Moderado"
    else:
        perfil = "Arrojado"

    return perfil, justificativa, score

investidores_encoded["perfil"] = investidores_encoded.apply(classificar_perfil_com_justificativa, axis=1)
print("\nDistribuição dos perfis (regras):")
print(investidores_encoded["perfil"].value_counts())


X = investidores_encoded.drop(columns=["perfil"])
y = investidores_encoded["perfil"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nANTES DO SMOTE:", y_train.value_counts().to_dict())
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("DEPOIS DO SMOTE:", y_train_bal.value_counts().to_dict())


rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight={"Arrojado": 3, "Moderado": 1, "Conservador": 1}
)
rf.fit(X_train_bal, y_train_bal)


y_pred = rf.predict(X_test)
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred, zero_division=0))

joblib.dump(rf, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nModelo e scaler salvos!")


investidores_encoded.to_csv("investidores_preprocessados.csv", index=False)









