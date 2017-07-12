const KNN = require('ml-knn')
const csv = require('csvtojson')
const prompt = require('prompt')
const {
  splitAt
} = require('ramda')

const csvPath = 'data/iris.csv'
const headers = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'type']

const separationMultiplier = 0.7

const shuffle = array => {
  let index = array.length
  while (index--) {
    const randomIndex = Math.floor(Math.random() * (index + 1))
    ;[array[index], array[randomIndex]] = [array[randomIndex], array[index]]
  }
  return array
}

function parse () {
  const data = []
  return new Promise(resolve =>
    csv({noheader: true, headers})
      .fromFile(csvPath)
      .on('json', json => data.push(json))
      .on('done', error => resolve(shuffle(data)))
  )
}

function dress (data) {
  const separate = splitAt(separationMultiplier * data.length)
  const types = Array.from(new Set(data.map(row => row.type)))
  const [x, y] = data.reduce((arrays, row) => {
    arrays[0].push(Object.keys(row).map(key => parseFloat(row[key])).slice(0, 4))
    arrays[1].push(types.indexOf(row.type))
    return arrays
  }, [[], []])
  const [trainingSetX, testSetX] = separate(x)
  const [trainingSetY, testSetY] = separate(y)
  return [trainingSetX, trainingSetY, testSetX, testSetY]
}

function train ([trainingSetX, trainingSetY, testSetX, testSetY]) {
  return [new KNN(trainingSetX, trainingSetY, {k: 7}), testSetX, testSetY]
}

function countErrors (predictions, expectations) {
  return predictions.reduce((count, prediction, index) =>
    count + Number(prediction !== expectations[index]), 0)
}

function test ([knn, testSetX, testSetY]) {
  const result = knn.predict(testSetX)
  const misclasses = countErrors(result, testSetY)
  console.log(`Test set size = ${testSetX.length}, misclassifications = ${misclasses}`)
}

function predict ([knn]) {
  prompt.start()
  prompt.get(headers.slice(0, 4), (error, result) => {
    if (error) throw error
    const values = Object.values(result).map(parseFloat)
    console.log(`with ${values}, type is ${knn.predict(values)}`)
  })
}

const knn = parse().then(dress).then(train)
knn.then(test)
knn.then(predict)
