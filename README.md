# ANÁLISE VISUAL E PREDITIVA DE DADOS DA VIOLÊNCIA NO BRASIL

Este projeto foi desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) em Gestão da Tecnologia da Informação pelo Instituto Federal do Sudeste de Minas Gerais (IF Sudeste MG - Campus Muriaé), sob a orientação do Professor Gustavo Willan Pereira.

Criadora
Flavia Santos - GitHub

Sobre o Projeto
A plataforma "Análise da Violência no Brasil" é uma ferramenta de Business Intelligence (BI) e Machine Learning que transforma dados governamentais brutos sobre segurança pública em uma aplicação web interativa. O sistema visa democratizar o acesso à informação, permitindo que qualquer usuário possa explorar visualmente as estatísticas de violência no Brasil (2015-2024) e utilizar um modelo de Inteligência Artificial para gerar previsões de tendências futuras.

Funcionalidades Principais
📊 Dashboard Interativo: Visualização de dados com filtros dinâmicos por Ano, Estado, Cidade e Tipo de Evento, através de gráficos de barras, linhas e pizza.

📜 Análise de Palavras: Geração de nuvens de palavras para identificar visualmente os tipos de ocorrências criminais mais frequentes em todo o período.

🧠 Módulo de Previsão: Utiliza um modelo de rede neural LSTM treinado para estimar o número de vítimas em anos futuros, com base em cenários personalizáveis pelo usuário.

📄 Páginas Informativas: Seções dedicadas a explicar a metodologia, as tecnologias utilizadas e a fonte dos dados do projeto.

Tecnologias Utilizadas:
Linguagem: Python

Aplicação Web: Streamlit

Análise de Dados: Pandas, NumPy

Visualização de Dados: Plotly, Matplotlib, WordCloud

Machine Learning: TensorFlow (Keras), Scikit-learn

Versionamento: Git, GitHub

Acessando a Aplicação Online
Este projeto está publicado como uma aplicação web e pode ser acessado diretamente pelo seu navegador, sem a necessidade de nenhuma instalação.

Acesse a ferramenta pelo link:
https://dados-violencia-brasil-2015-a-2024.streamlit.app/

Como Usar a Ferramenta
A aplicação é dividida em módulos, que podem ser acessados pela barra de navegação lateral.

📊 Dashboard de Análise
Utilize os filtros no topo da página (Ano, Estado, Tipo de Evento) para segmentar os dados.

Ao selecionar um único estado, um filtro de Cidades será habilitado para uma análise mais detalhada.

Os gráficos e a tabela de dados serão atualizados automaticamente de acordo com sua seleção.

📜 Análise de Palavras
Esta seção exibe uma nuvem de palavras com a frequência dos tipos de evento em todo o período analisado.

Abaixo da nuvem, há uma tabela expansível com os valores quantitativos de cada evento.

🧠 Módulo de Previsão
Clique em "Iniciar Nova Previsão" para abrir o formulário.

Preencha o ano futuro desejado e, opcionalmente, refine o cenário com os filtros de UF, Cidade, Evento, etc.

Clique em "Calcular Estimativa" para que o modelo de IA processe a solicitação e exiba o resultado.

Contato
Se tiver dúvidas ou sugestões, entre em contato: [coloque seu e-mail aqui]
