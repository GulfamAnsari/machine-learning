let INPUT_SIZE = 1200;
let cache = null;
const mock = true;
const API_KEY = "3FYXCDC9TTNBASNX";
// const API_KEY = "E19PE4N9AC02MRWZ";


const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const MOCK = require("./mock");

const app = express();
const PORT = 5000;
app.use(express.json());


// API endpoint to get predictions
app.get("/predict/:symbol", async (req, res) => {
  const stockSymbol = req.params.symbol;
  const data = await fetchStockData(stockSymbol);
  console.log("Done");
  res.json({ ...data });
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));


async function fetchStockData(symblo) {
  const stockSymbol = symblo.toUpperCase();
  if (!stockSymbol) {
    alert("Please enter a stock symbol.");
    return;
  }
  console.log("Fetching 10 years of data...");

  const url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=${stockSymbol}.BSE&outputsize=full&apikey=${API_KEY}`;

  try {
    let data = null;
    if (mock) {
        data = MOCK;
    } else {
        const response = await fetch(url);
        data = await response.json();
    }
    
    if (!data["Time Series (Daily)"])
      throw new Error("Invalid stock symbol or API limit exceeded.");

    // Process data: Convert to array and sort by date (oldest to newest)
    const prices = Object.entries(data["Time Series (Daily)"])
      .map(([date, value]) => ({
        date,
        price: parseFloat(value["4. close"])
      }))
      .sort((a, b) => new Date(a.date) - new Date(b.date)); // Sort in ascending order
    const pData = await trainAndPredict(prices.slice(-(INPUT_SIZE)));
    return pData;
  } catch (error) {
    console.error(error);
  }
}

async function trainAndPredict(data) {
    console.log("Total Data Points:", data.length);
  
    if (data.length < 1000) {
      console.error(`🚨 Not enough data. At least 367 data points required, but got only ${data.length}.`);
      return { data, futurePrices: [] };
    }
  
    console.log("Training model...");
  
    // Extract prices and normalize
    const prices = data.map((d) => d.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const range = maxPrice - minPrice;
  
    const normalizedPrices = prices.map((p) => (p - minPrice) / range);
  
    // Prepare training data
    const inputSize = 360; // Use past 360 days
    const outputSize = 30;  // Predict next 7 days
    const xs = [];
    const ys = [];
  
    for (let i = 0; i < normalizedPrices.length - inputSize - outputSize; i++) {
      const inputSlice = normalizedPrices.slice(i, i + inputSize);
      
      // Ensure correct shape for TensorFlow tensor3d()
      const formattedInput = inputSlice.map((v) => [v]);
  
      xs.push(formattedInput);
      ys.push(normalizedPrices.slice(i + inputSize, i + inputSize + outputSize));
    }
  
    if (xs.length === 0 || ys.length === 0) {
      console.error("🚨 Training data is empty after processing.");
      return { data, futurePrices: [] };
    }
  
    console.log(`Training on ${xs.length} samples.`);
  
    // Convert to tensors
    const tensorXs = tf.tensor3d(xs, [xs.length, inputSize, 1]);
    const tensorYs = tf.tensor2d(ys, [ys.length, outputSize]);
  
    // Define the LSTM model
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 64, returnSequences: false, inputShape: [inputSize, 1] }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: outputSize }));
  
    model.compile({ optimizer: tf.train.adam(0.001), loss: "meanSquaredError" });
  
    console.log("Training model...");
    await model.fit(tensorXs, tensorYs, {
      epochs: 100,
      batchSize: 16,
      verbose: 1,
      validationSplit: 0.1,
      callbacks: tf.callbacks.earlyStopping({ monitor: "val_loss", patience: 5 }),
    });
  
    tensorXs.dispose();
    tensorYs.dispose();
  
    console.log("Preparing data for prediction...");
    const last360Days = normalizedPrices.slice(-inputSize).map((v) => [v]);
    const tensorInput = tf.tensor3d([last360Days], [1, inputSize, 1]);
  
    try {
      console.log("Running prediction...");
      const predictionTensor = model.predict(tensorInput);
  
      if (!predictionTensor) {
        throw new Error("🚨 Model prediction returned undefined.");
      }
  
      console.log("Prediction Tensor:", predictionTensor);
  
      const predictedNormalized = await predictionTensor.data();
      predictionTensor.dispose();
      tensorInput.dispose();
  
      // Convert back to original price range
      const futurePrices = predictedNormalized.map((p) => p * range + minPrice);
      console.log("✅ Future Prices:", futurePrices);
      return { data, futurePrices };
    } catch (error) {
      console.error("🚨 Prediction error:", error);
      return { data, futurePrices: [] };
    }
  }
  