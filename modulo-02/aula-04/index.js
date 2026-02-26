import tf from '@tensorflow/tfjs-node';

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // Primeira camada da rede neural:
    // entrada de 7 posições {idade, cor (3), localização (3)}

    // 80 neuronios: aqui tem tudo isso porque tem pouca base de treino
    // quanto mais neuronios, mais complexidade a rede pode aprender
    // e consequentemente, mais processamento ela vai usar

    // a ReLU age como filtro:
    // é como se ela deixasse somente os dados interessantes seguirem viagem na rede
    // Se a informação chegou nesse neuronio é positiva, passa para frente!
    // Se for zero ou negativa, joga fora, não vai servir para nada
    model.add(tf.layers.dense({
        inputShape: [7], units: 80, activation: 'relu'
    }))

    // Saída: 3 neuronios - porque tem 3 categorias (premium, medium, basic)
    // activation softmax: normaliza a saída em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // Compirando o modelo
    // optimizer Adam (Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais
    // ajusta os pesos de forma eficiente e inteligente
    // aprender com histórico de erros e acertos

    // loss: categoricalCrossentropy
    // Ele compara o que o modelo "acha" (os scores de cada categoria)
    // com o que é a resposta correta (labels)
    // a categoria premium sera sempre [1, 0, 0], medium [0, 1, 0], basic [0, 0, 1]

    // quanto mais distante da previsao do modelo da resposta correta
    // maior o erro (loss)
    // exemplo classico: classificacao de imagens, recomendacao, categorizacao de usuario, etc
    // qualquer coisa em que a resposta certa é apenas uma entre varias possiveis
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] })

    // Treinamento do modelo
    await model.fit(inputXs, outputYs, {
        verbose: 0, // nao mostrar detalhes do treinamento
        epochs: 100, // numero de vezes que o modelo vai passar por todo o dataset de treino
        shuffle: true, // embaralha os dados a cada epoca para evitar que o modelo aprenda a ordem dos dados
        // callbacks: {
        //     onEpochEnd: (epoch, log) => console.log(
        //         `Epoch ${epoch}: loss = ${log.loss}`
        //     )
        // }
    })
    return model
}

async function predict(model, pessoa) {
    // transformar o array js para o tensor do tensorflow
    const tfInput = tf.tensor2d(pessoa)

    // fazer a previsao usando o modelo treinado (output sera um vetor de 3 probabilidades)
    const pred = model.predict(tfInput)
    const predArray = await pred.array() // converter o tensor de volta para array js
    return predArray[0].map((prob, index) => ({ prob, index }))
}

const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: "Zé", cor: "verde", localizacao: "Curitiba", idade: 28 }
// normalizando a idade da nova pessoa usando o mesmo padrao do treino
// Exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2
const pessoaTensorNormalizada = [
    [
        0.2, // idade normalizada
        0,   // azul
        0,   // vermelho
        1,   // verde
        0,   // São Paulo
        0,   // Rio
        1    // Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizada)
const results = predictions
    .sort((a, b) => b.prob - a.prob)// ordenar por probabilidade decrescente
    .map(p => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`) // mapear para nome da categoria e formatar a probabilidade
    .join('\n') // juntar tudo em uma string com quebras de linha

console.log(results);
