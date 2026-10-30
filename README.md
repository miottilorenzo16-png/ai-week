# Scenario Lens

An educational portfolio stress-testing prototype built for a one-day AI and quantitative finance hackathon.

## What it does

- Starts with a fictional five-holding portfolio.
- Lets you compare three illustrative market scenarios: an energy price spike, a 1 percentage-point rate rise, and a growth slowdown.
- Shows the weighted impact of each holding and an explanation of the largest contributors.
- Lets you adjust portfolio weights while keeping the total at 100%.

## Run it

Open [index.html](./index.html) in a browser. There are no dependencies, build tools, accounts, or API keys.

## Important limits

The scenario shocks are invented example assumptions, not historical estimates. The app calculates a simplified weighted sum:

`portfolio impact = Σ (holding weight × assumed holding shock)`

It does not use live market data, forecast returns, or provide investment advice.

## Next hackathon improvements

1. Replace example shocks with transparent historical scenarios and cite the data source.
2. Add a small AI explanation layer that receives only the already-calculated results and assumptions.
3. Add a simple chart comparing the scenarios.
