from flask import Flask, request, jsonify
from flask_cors import CORS
from recomenda import recomendar_investimentos  # função externa

app = Flask(__name__)
CORS(app)

@app.route("/recomendar", methods=["POST"])
def recomendar():
    dados = request.get_json()

    try:
       
        perfil, justificativas, top_df = recomendar_investimentos(dados)


        top_investimentos = top_df.to_dict(orient="records")

       
        descricao = ""
        if perfil == "Conservador":
            descricao = "Você prefere segurança e estabilidade nos investimentos."
        elif perfil == "Moderado":
            descricao = "Você busca equilíbrio entre risco e retorno."
        else: 
            descricao = "Você está disposto a correr mais riscos em busca de maiores retornos. Investimentos muito diversificados e de maior potencial de rentabilidade foram selecionados."

        return jsonify({
            "perfil": perfil,
            "descricao": descricao,
            "justificativas": justificativas,
            "top_investimentos": top_investimentos
        })

    except Exception as e:
        return jsonify({"error": f"Erro ao gerar recomendação: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
