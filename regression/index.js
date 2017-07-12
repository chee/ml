const ml = require('ml-regression')
const csv = require('csvtojson')
const SLR = ml.SLR
const readline = (require('readline')).createInterface({
  input: process.stdin,
  output: process.stdout
})

const csvFilePath = 'data/ads.csv'

let parsedData = []
let input = []
let output = []
let regressionModel

function dressData () {
  parsedData.forEach(row => {
    input.push(parseFloat(row.Radio))
    output.push(parseFloat(row.Sales))
  })
}

function predictOutput () {
  readline.question('enter input for prediction: ', answer => {
    console.log(`at input ${answer}, output will be ${regressionModel.predict(parseFloat(answer))}`)
    predictOutput()
  })
}

function performRegression () {
  regressionModel = new SLR(input, output)
  console.log(regressionModel.toString(3))
  predictOutput()
}

csv()
  .fromFile(csvFilePath)
  .on('json', json => parsedData.push(json))
  .on('done', () => {
    dressData()
    performRegression()
  })
