
const { Alchemy, Network } = require("alchemy-sdk");
const fs = require("fs");
const csv = require("csv-parser");

const config = {
  apiKey: "-oE3ZwQi9-Fkxj9kO-pETWTHNT4WX2NR",
  network: Network.ETH_MAINNET,
};
const alchemy = new Alchemy(config);

function fetchAndSaveTransactions(address) {
  alchemy.core
    .getAssetTransfers({
      fromBlock: "0x0",
      fromAddress: address,
      category: ["external", "internal", "erc20", "erc721", "erc1155"],
    })
    .then((data) => {
      const result = { [address]: data };
      const resultJson = JSON.stringify(result, null, 2);

      // Write result to a JSON file
      fs.writeFileSync("dataset/alchemy_user_transactions.json", resultJson, { flag: "a" });

      console.log(`Transactions for address ${address} saved.`);
    })
    .catch((error) => {
      console.error(`Error occurred while fetching transactions for address ${address}`);
      console.error(error);
    });
}

// Read unique addresses from CSV and call fetch function for each address
fs.createReadStream("dataset/eoa_addresses.csv")
  .pipe(csv())
  .on("data", (row) => {
    const address = row["address"];
    fetchAndSaveTransactions(address);
  })
  .on("end", () => {
    console.log("Fetching and saving transactions completed.");
  });