import databento as db
import pandas as pd

client = db.Historical("db-LEnwVehwtuhQu8KmU3uFDLJ8V3Xhm")
# Set some common args
dataset = "GLBX.MDP3"
# First 30 minutes from open
start = "2023-04-03T13:00:00"
end = "2023-04-03T13:10:00"

# First, get the instrument ID of front month future
front_fut_res = client.symbology.resolve(
    symbols=["ES.n.0"],
    stype_in="continuous",
    stype_out="instrument_id",
    dataset="GLBX.MDP3",
    start_date="2023-04-03",
)
front_fut_id = int(front_fut_res["result"]["ES.n.0"][0]["s"])

# Then GET all option definitions, starting from Sunday
opt_defs = client.timeseries.get_range(
    dataset=dataset,
    schema="definition",
    symbols=["LO.OPT"],
    stype_in="parent",
    start="2023-04-02",
).to_df()
# Filter to front month outrights
front_opt_defs = opt_defs.loc[
    (opt_defs.user_defined_instrument == "N")
    & (opt_defs.instrument_class.isin({"C", "P"}))
    & (opt_defs.underlying_id == front_fut_id),
    :,
]
# Then get future prices
fut_prices = client.timeseries.get_range(
    dataset=dataset,
    schema="mbp-1",
    symbols=front_fut_id,
    stype_in="instrument_id",
    start=start,
    end=end,
).to_df()
# And option trades
opt_prices = client.timeseries.get_range(
    dataset=dataset,
    schema="trades",
    symbols=front_opt_defs.instrument_id.unique().tolist(),
    stype_in="instrument_id",
    start=start,
    end=end,
).to_df()

# Join options with their definitions
combined = opt_prices.merge(
    opt_defs,
    on="instrument_id",
    how="inner",
    suffixes=("", "_def"),
).set_index("ts_event")
# Finally join options with their underlying prices
combined[["underlying_ask", "underlying_bid"]] = combined.apply(
    lambda r: fut_prices.loc[: r.name, ["ask_px_00", "bid_px_00"]].iloc[-1],
    axis=1,
)
# Don't abbreviate columns
pd.set_option("display.max_columns", None)
print(
    combined[
        ["strike_price", "instrument_class", "price", "underlying_ask", "underlying_bid"]
    ].head(),
)