# pipeline_bl_risk_parity_erc_with_growth.py

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import logging # Import logging

from sklearn.preprocessing import MinMaxScaler
from fredapi import Fred
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt import EfficientFrontier
# Import necessary pypfopt modules for ERC and risk models
from pypfopt.risk_parity_optimizer import RiskParityOptimizer
from pypfopt import risk_models


# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Configuration ---
# FRED API key for GDP growth
FRED_API_KEY = "f9d5154112088b889515fb93fabc8ccd" # Replace with your key if different
fred = Fred(api_key=FRED_API_KEY)

DATA_FOLDER = "." # CSVファイル配置フォルダ
TICKERS = [
    "VTI","EWG","EWQ","EWJ","FXI","INDA",
    "EWY","EWZ","KSA","EZA","EWA","EWC"
]
MARKET_WEIGHTS = {
    "VTI":0.58, "EWG":0.03, "EWQ":0.025, "EWJ":0.06,
    "FXI":0.04, "INDA":0.03, "EWY":0.02, "EWZ":0.02,
    "KSA":0.015, "EZA":0.01, "EWA":0.03, "EWC":0.03
}
# Base Q_MAP (expected excess returns)
Q_MAP = {
    "VTI":0.00610, "EWG":0.00920, "EWQ":0.00550, "EWJ":-0.00380,
    "FXI":0.00520, "INDA":0.00510, "EWY":0.00320, "EWZ":-0.00650,
    "KSA":0.01100, "EZA":-0.00820, "EWA":0.00740, "EWC":0.00940
}
# FRED GDP growth symbols (annual % change)
GDP_SYMBOLS = {
    "VTI": "A191RL1Q225SBEA",    # US
    "EWG": "CLVMNACSCAB1GQDE",   # DE
    "EWQ": "CLVMNACSCAB1GQFR",   # FR
    "EWJ": "JPNRGDPRPCHPT",      # JP
    "FXI": "CHRGDPNQDSMEI",      # CN
    "INDA":"INDRGDPNQDSMEI",     # IN
    "EWY": "KRRGDPNQDSMEI",      # KR
    "EWZ": "BRRGDPNQDSMEI",      # BR
    "KSA": None, # Data not readily available or symbol unknown
    "EZA": None, # Data not readily available or symbol unknown
    "EWA": None, # Data not readily available or symbol unknown
    "EWC": None  # Data not readily available or symbol unknown
}

DELTA = 2.5
TAU   = 0.05
COST  = 0.001  # 0.1% transaction cost (applied per total turnover)


# --- Load Prices ---
def load_price_data(folder):
    """Loads price data from CSV files in the specified folder."""
    logging.info(f"Loading price data from {folder}")
    dfs = []
    for t in TICKERS:
        file_path = os.path.join(folder, f"{t}.csv")
        if not os.path.exists(file_path):
            logging.warning(f"Price data file not found for ticker {t}: {file_path}")
            continue # Skip if file doesn't exist
        try:
            # Assuming price is in the first column
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if df.empty:
                logging.warning(f"Price data file is empty for ticker {t}: {file_path}")
                continue # Skip if file is empty
            dfs.append(df.iloc[:,0].rename(t))
        except Exception as e:
            logging.error(f"Error loading price data for ticker {t} from {file_path}: {e}")
            continue # Skip if error occurs

    if not dfs:
        logging.error("No valid price data loaded for any ticker.")
        return pd.DataFrame() # Return empty DataFrame if no data

    combined_df = pd.concat(dfs, axis=1).sort_index()

    # Optional: Forward fill missing data if needed, or handle NaNs later
    # combined_df = combined_df.ffill()

    logging.info(f"Successfully loaded price data for {len(combined_df.columns)} tickers.")
    return combined_df


# --- Value Factor via Stooq (P/B & P/E) ---
def fetch_value_from_stooq(tickers):
    """Fetches latest P/B and P/E data for tickers from Stooq."""
    logging.info("Attempting to fetch latest value data from Stooq...")
    data = {}
    for t in tickers:
        sym = t.lower()+".us"
        url_pb = f"https://stooq.com/q/d/l/?s={sym}&i=m&f=pb"
        url_pe = f"https://stooq.com/q/d/l/?s={sym}&i=m&f=pe"
        try:
            # Add headers to mimic browser and potentially avoid blocking
            headers = {'User-Agent': 'Mozilla/5.0'}
            # Using requests might be more robust than pd.read_csv directly for URLs
            # import requests
            # response_pb = requests.get(url_pb, headers=headers)
            # df_pb = pd.read_csv(StringIO(response_pb.text), parse_dates=["Date"], index_col="Date")
            df_pb = pd.read_csv(url_pb, parse_dates=["Date"], index_col="Date")
            df_pe = pd.read_csv(url_pe, parse_dates=["Date"], index_col="Date")

            latest_pb = df_pb["Close"].iloc[-1] if not df_pb.empty else np.nan
            latest_pe = df_pe["Close"].iloc[-1] if not df_pe.empty else np.nan

            data[t] = {"PB": latest_pb, "PE": latest_pe}
            logging.debug(f"Fetched Stooq data for {t}") # Use debug for verbose info
        except Exception as e:
            data[t] = {"PB": np.nan, "PE": np.nan}
            logging.warning(f"Could not fetch Stooq data for {t}: {e}")

    df = pd.DataFrame(data).T
    logging.info("Finished fetching latest value data from Stooq.")
    return df

def compute_value_score(df):
    """Computes a value score based on P/B and P/E, scaled between 0 and 1."""
    logging.info("Computing value scores.")
    # Fill NaNs strategically. Filling with max might penalize assets with no data.
    # Consider filling with median or a neutral value (like 0.5 after scaling)
    # For now, keep original logic but add a check
    if df.isnull().all().all():
        logging.warning("All value data is NaN. Returning neutral score 0.5 for all.")
        return pd.Series(0.5, index=df.index, name="Value_Score")

    # Fill remaining NaNs (e.g., only some tickers are NaN) with column median or mean
    df_filled = df.fillna(df.median()) # Using median might be safer than max

    if df_filled.isnull().any().any():
         logging.warning("NaNs still present in value data after filling with median.")


    scaler = MinMaxScaler()
    score_pb = np.full((len(df_filled),1),0.5) # Default in case of scaler error
    score_pe = np.full((len(df_filled),1),0.5) # Default in case of scaler error

    try:
        # Check if column has variance, otherwise scaler will raise error
        if df_filled["PB"].nunique() > 1:
            score_pb = 1 - scaler.fit_transform(df_filled[["PB"]])
        else:
            logging.warning("PB data is constant. Cannot scale, using score 0.5.")
    except Exception as e:
        logging.error(f"Error scaling PB data: {e}. Using score 0.5.")


    try:
        if df_filled["PE"].nunique() > 1:
            score_pe = 1 - scaler.fit_transform(df_filled[["PE"]])
        else:
            logging.warning("PE data is constant. Cannot scale, using score 0.5.")
    except Exception as e:
        logging.error(f"Error scaling PE data: {e}. Using score 0.5.")


    vs = (score_pb + score_pe) / 2
    score_series = pd.Series(vs.flatten(), index=df_filled.index, name="Value_Score")
    logging.info("Value scores computed.")
    return score_series


# --- Growth Factor via FRED ---
def fetch_growth_data(tickers):
    """Fetches the latest GDP growth data for tickers from FRED."""
    logging.info("Attempting to fetch latest growth data from FRED...")
    data = {}
    for t in tickers:
        sym = GDP_SYMBOLS.get(t)
        if sym:
            try:
                series = fred.get_series(sym)
                if series is not None and not series.empty:
                    growth = series.iloc[-1]
                    logging.debug(f"Fetched FRED data for {t} ({sym}): {growth}")
                else:
                    growth = np.nan
                    logging.warning(f"FRED series is empty or None for {t} ({sym}).")
            except Exception as e:
                growth = np.nan
                logging.warning(f"Could not fetch FRED data for {t} ({sym}): {e}")
        else:
            growth = np.nan
            logging.debug(f"No FRED symbol defined for {t}. Setting growth to NaN.")

        data[t] = growth

    series = pd.Series(data, name="GDP_Growth")
    logging.info("Finished fetching latest growth data from FRED.")
    return series

def compute_growth_score(series):
    """Computes a growth score based on GDP growth, scaled between 0 and 1."""
    logging.info("Computing growth scores.")
    df = series.to_frame()

    if df["GDP_Growth"].isnull().all():
        logging.warning("All growth data is NaN. Returning neutral score 0.5 for all.")
        return pd.Series(0.5, index=df.index, name="Growth_Score")

    # Fill NaNs with median for scaling
    median = df["GDP_Growth"].median()
    df["GDP_Growth"] = df["GDP_Growth"].fillna(median)

    if df["GDP_Growth"].nunique() <= 1:
        logging.warning("Growth data is constant or only one non-NaN value. Cannot scale, using score 0.5.")
        score = np.full((len(df),1),0.5)
    else:
        scaler = MinMaxScaler()
        try:
            score = scaler.fit_transform(df)
        except Exception as e:
            logging.error(f"Error scaling growth data: {e}. Using score 0.5.")
            score = np.full((len(df),1),0.5)


    score_series = pd.Series(score.flatten(), index=df.index, name="Growth_Score")
    logging.info("Growth scores computed.")
    return score_series


# --- Factor Pre-fetching (Modified) ---
# NOTE: This fetches *latest* factors once. For historical backtest,
# this needs to be replaced with logic that fetches/looks up historical factors
# corresponding to each rebalance date.
def fetch_all_required_factor_data(tickers):
    """
    Fetches factor data (Value, Growth) for the given tickers.
    Currently fetches only the latest data.
    """
    logging.info("Fetching initial factor data.")
    factor_data = {}

    # --- Fetch and Compute Value Score ---
    df_val = fetch_value_from_stooq(tickers)
    factor_data['value_score'] = compute_value_score(df_val)

    # --- Fetch and Compute Growth Score ---
    series_growth = fetch_growth_data(tickers)
    factor_data['growth_score'] = compute_growth_score(series_growth)

    logging.info("Finished fetching initial factor data.")
    # Return the computed factor scores as pandas Series
    return factor_data


# --- ERC Solver (Removed - using pypfopt) ---
# The original compute_erc_weights function is removed.
# We will use pypfopt's RiskParityOptimizer.

# --- Compute Weights: BL + Value + Growth + ERC/Fallback (Modified) ---
def compute_rp_bl_weights(price_df_slice, factor_scores): # Pass factor scores
    """
    Computes portfolio weights using Black-Litterman, Value/Growth factors,
    and falls back to ERC, Market Cap, or Equal Weight if optimization fails.
    """
    if price_df_slice.empty:
        logging.error("Price slice is empty. Cannot compute weights.")
        return {} # Return empty weights

    logging.info("Calculating covariance matrix.")
    # Use min_periods for pct_change to avoid NaNs at the start of slice if needed
    # Drop NaNs resulting from pct_change before cov
    returns = price_df_slice.pct_change().dropna()
    if returns.empty:
        logging.warning("Returns data is empty after dropping NaNs. Cannot compute covariance.")
        # Fallback to equal weights immediately if no return data
        assets = price_df_slice.columns.tolist()
        eq_w = 1.0 / len(assets) if assets else 0
        weights = {a: eq_w for a in assets}
        logging.warning("Falling back to Equal Weights due to no returns data.")
        return weights


    try:
        # Use pypfopt's risk_models for covariance
        cov = risk_models.sample_cov(returns) * 252 # Annualize
        logging.info("Covariance matrix computed.")
    except Exception as e:
        logging.error(f"Error computing covariance matrix: {e}.")
        # Fallback to equal weights if covariance fails
        assets = price_df_slice.columns.tolist()
        eq_w = 1.0 / len(assets) if assets else 0
        weights = {a: eq_w for a in assets}
        logging.warning("Falling back to Equal Weights due to covariance error.")
        return weights


    assets = cov.columns.tolist() # Use assets from cov as some might have been dropped

    # Get pre-fetched factor scores
    val_score = factor_scores['value_score']
    grow_score = factor_scores['growth_score']

    # Mixed Q: base 70% + value 10% + growth 20%
    # Ensure factors are accessible by asset ticker and use default if missing
    mixed_Q = {
        a: 0.7*Q_MAP.get(a, 0)
           + 0.1*val_score.get(a, 0.5) # Default to 0.5 if score missing for asset
           + 0.2*grow_score.get(a, 0.5) # Default to 0.5 if score missing for asset
        for a in assets # Iterate over assets in cov
    }

    # π (implied returns from market weights)
    # Ensure market weights align with assets in cov
    w_mkt = np.array([MARKET_WEIGHTS.get(a, 0) for a in assets])
    if np.sum(w_mkt) == 0:
        logging.warning("Sum of market weights for available assets is zero. Implied returns (pi) will be zero.")
    try:
        # Ensure consistent types/shapes for dot product
        pi = DELTA * cov.values.dot(w_mkt)
        # Convert pi back to pandas Series for consistency with pypfopt
        pi_series = pd.Series(pi, index=assets)
    except Exception as e:
        logging.error(f"Error computing pi: {e}. Falling back to pi=0.")
        pi_series = pd.Series(0.0, index=assets) # Fallback for pi

    # Q (views)
    Q_vec = np.array([mixed_Q.get(a, 0) for a in assets])
    Q_series = pd.Series(Q_vec, index=assets) # Convert Q to pandas Series

    # Omega (uncertainty in views) - Use pypfopt's risk model
    try:
        # Using diag(tau*cov) as in original logic for Omega structure
        # Note: Ledoit-Wolf was an example, sticking closer to original logic here
        omega = np.diag(np.diag(TAU * cov.values))
        omega_df = pd.DataFrame(omega, index=assets, columns=assets) # Convert to DataFrame
        logging.debug("Omega matrix computed.")
    except Exception as e:
        logging.error(f"Error computing omega: {e}. Cannot perform Black-Litterman.")
        # If omega fails, BL cannot be computed, skip to next fallback
        post_series = pi_series # Use pi as the 'posterior' if omega fails


    # BL posterior returns
    post_series = pi_series # Default posterior is pi if BL fails
    try:
        # Ensure inputs are pandas objects for BlackLittermanModel
        bl = BlackLittermanModel(cov, pi_series, P=np.eye(len(assets)), Q=Q_series, omega=omega_df, tau=TAU)
        post_series = bl.bl_returns()
        logging.info("Black-Litterman returns computed successfully.")
    except Exception as e:
        logging.error(f"Black-Litterman calculation failed: {e}. Using pi (implied returns) instead as posterior.")
        post_series = pi_series # Use pi if BL fails


    # Optimization cascade
    weights = None

    # 1) Max Sharpe
    logging.info("Attempting Max Sharpe optimization.")
    try:
        # Ensure post and cov are pandas objects
        ef = EfficientFrontier(post_series, cov)
        # Add constraints (no shorting, weights sum to 1)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: np.sum(w) == 1)

        ef.max_sharpe()
        weights = ef.clean_weights()
        logging.info("Max Sharpe optimization successful.")
        return weights
    except Exception as e:
        logging.warning(f"Max Sharpe optimization failed: {e}. Trying ERC.")


    # 2) ERC using pypfopt
    logging.info("Attempting ERC optimization.")
    try:
        # Ensure cov is pandas DataFrame
        rp = RiskParityOptimizer(cov)
        # Add constraints (no shorting, weights sum to 1)
        rp.add_constraint(lambda w: w >= 0)
        rp.add_constraint(lambda w: np.sum(w) == 1)

        raw_weights = rp.optimize_erc()
        # pypfopt's optimize_erc returns a dictionary
        weights = rp.clean_weights() # Clean weights adds a small tolerance
        logging.info("ERC optimization successful.")
        return weights
    except Exception as e:
        logging.warning(f"ERC optimization failed: {e}. Trying Market Weights.")


    # 3) Market weights
    logging.info("Attempting Market Weights fallback.")
    try:
        # Ensure market weights align with assets in cov
        weights = {a: MARKET_WEIGHTS.get(a, 0) for a in assets}
        # Normalize market weights for the available assets
        total_mkt_weight = sum(weights.values())
        if total_mkt_weight > 0:
            weights = {a: w / total_mkt_weight for a, w in weights.items()}
        else:
            raise ValueError("Sum of market weights for available assets is zero.") # Trigger fallback

        logging.info("Falling back to Normalized Market Weights.")
        return weights
    except Exception as e:
        logging.warning(f"Market Weights fallback failed: {e}. Trying Equal Weights.")


    # 4) Equal weight
    logging.info("Attempting Equal Weights fallback.")
    assets = cov.columns.tolist() # Use assets from cov
    eq = 1.0 / len(assets) if assets else 0 # Avoid division by zero
    weights = {a: eq for a in assets}
    if not assets:
        logging.warning("No assets available. Returning empty weights.")
    else:
        logging.warning("Falling back to Equal Weights.")

    return weights


# --- Backtest ---
def generate_rebalance_dates(price_df):
    """Generates quarterly rebalance dates (Jan, Apr, Jul, Oct) from price data index."""
    if price_df.empty:
        return []
    # Ensure start and end dates are included in the potential range
    start_date = price_df.index[0].normalize()
    end_date = price_df.index[-1].normalize()

    # Generate potential rebalance dates (1st day of Jan, Apr, Jul, Oct)
    dates = pd.date_range(start=start_date.to_period('Q').to_timestamp(),
                          end=end_date.to_period('Q').to_timestamp() + pd.offsets.QuarterEnd(0),
                          freq='QS-JAN') # Quarterly start, starting Jan

    # Filter dates to be within the price data range
    rebalance_dates = [d for d in dates if d >= start_date and d <= end_date]

    # Ensure the very first date of price data is not a rebalance date
    # as we need previous day price for return calculation.
    # The first rebalance typically happens *after* the strategy starts.
    # Let's find the first potential rebalance date *after* the data start.
    first_data_date = price_df.index[0]
    adjusted_rebalance_dates = [d for d in rebalance_dates if d > first_data_date]

    # Optional: Add the very last date if it's a rebalance date and within data range
    # if end_date in rebalance_dates and end_date not in adjusted_rebalance_dates:
    #     adjusted_rebalance_dates.append(end_date)

    return sorted(list(set(adjusted_rebalance_dates))) # Use set to remove duplicates and sort


def backtest(price_df, cost):
    """Runs the backtest simulation."""
    if price_df.empty:
        logging.error("Price data is empty. Cannot run backtest.")
        return pd.Series(dtype=float) # Return empty series

    nav     = pd.Series(index=price_df.index, dtype=float)
    w_curr = None # Current weights
    value   = 1.0 # Initial portfolio value

    # --- Pre-fetch factors once ---
    # NOTE: This uses the LATEST factors for ALL rebalances.
    # For a proper backtest, replace this with historical factor lookup per date.
    try:
        factor_scores = fetch_all_required_factor_data(price_df.columns.tolist())
        if not factor_scores or factor_scores.get('value_score') is None or factor_scores.get('growth_score') is None:
            logging.warning("Initial factor data fetch returned empty or invalid scores. Factors will default to 0.5 scores.")
            # Provide default neutral scores if fetching failed entirely
            assets = price_df.columns.tolist()
            factor_scores = {
                'value_score': pd.Series(0.5, index=assets),
                'growth_score': pd.Series(0.5, index=assets)
            }

    except Exception as e:
        logging.error(f"Error during initial factor data fetch: {e}. Factors will default to 0.5 scores.")
        # Provide default neutral scores if fetching failed entirely
        assets = price_df.columns.tolist()
        factor_scores = {
            'value_score': pd.Series(0.5, index=assets),
            'growth_score': pd.Series(0.5, index=assets)
        }


    rebalance_dates = generate_rebalance_dates(price_df)
    logging.info(f"Generated {len(rebalance_dates)} rebalance dates: {rebalance_dates}")

    for date in price_df.index:
        # Handle the first day of the backtest separately or ensure return calc starts from day 2
        if date == price_df.index[0]:
            nav.loc[date] = value
            logging.debug(f"Starting backtest on {date.strftime('%Y-%m-%d')}")
            continue # Move to the next day for return calculation

        # --- Rebalancing ---
        # Rebalance on scheduled dates OR if current weights are None (first rebalance opportunity)
        if date in rebalance_dates or w_curr is None:
            # Ensure rebalance happens only *after* the first day
            if date > price_df.index[0]:
                logging.info(f"Rebalancing on {date.strftime('%Y-%m-%d')}")
                try:
                    # Use trailing window (e.g., 252 trading days ~ 1 year) for covariance calculation
                    lookback_slice = price_df.loc[:date].iloc[-252:]
                    if len(lookback_slice) < 2: # Need at least 2 data points for returns/covariance
                        logging.warning(f"Not enough historical data for covariance on {date.strftime('%Y-%m-%d')}. Skipping rebalance.")
                        # Keep previous weights if any, otherwise fall back
                        if w_curr is None:
                            logging.warning("First rebalance failed due to insufficient data. Falling back to equal weight.")
                            assets = price_df.columns.tolist()
                            eq = 1.0 / len(assets) if assets else 0
                            w_curr = {a: eq for a in assets}
                        # If subsequent rebalances fail due to data, w_curr remains the previous valid weights
                    else:
                        # Pass the appropriate price slice and factor scores
                        w_new = compute_rp_bl_weights(lookback_slice, factor_scores)

                        # Calculate and apply transaction cost based on turnover
                        # Only apply cost if we are actually changing weights (i.e., w_curr exists and is different from w_new)
                        if w_curr is not None and w_new is not None: # Ensure both sets of weights exist
                            # Ensure weight dictionaries have same keys for turnover calculation
                            all_assets = set(w_new.keys()) | set(w_curr.keys())
                            turnover = sum(abs(w_new.get(a, 0) - w_curr.get(a, 0)) for a in all_assets) / 2
                            if turnover > 1e-6: # Apply cost only if turnover is significant
                                value *= (1 - cost * turnover)
                                logging.info(f"Applied transaction cost on {date.strftime('%Y-%m-%d')}. Turnover: {turnover:.4f}, Value after cost: {value:.4f}")
                            else:
                                logging.debug(f"Skipping transaction cost on {date.strftime('%Y-%m-%d')} due to negligible turnover.")


                        w_curr = w_new
                        logging.info(f"New weights computed for {date.strftime('%Y-%m-%d')}: {w_curr}")

                except Exception as e:
                    logging.error(f"Error during rebalancing on {date.strftime('%Y-%m-%d')}: {e}. Using previous weights if available.")
                    if w_curr is None:
                        # If the very first rebalance fails completely, fall back to equal weight
                        logging.warning("First rebalance failed completely. Falling back to equal weight.")
                        assets = price_df.columns.tolist()
                        eq = 1.0 / len(assets) if assets else 0
                        w_curr = {a: eq for a in assets}
                    # If subsequent rebalances fail, w_curr remains the previous valid weights
            else:
                # This case should ideally not be reached if logic is correct,
                # but included for safety if date == price_df.index[0] falls into rebalance_dates
                logging.debug(f"Skipping rebalance on {date.strftime('%Y-%m-%d')} as it's the first data point.")


        # --- Portfolio Value Update ---
        if w_curr is None:
            # Should not happen if first rebalance has fallback, but defensive
            logging.warning(f"No weights available on {date.strftime('%Y-%m-%d')}. NAV unchanged.")
            nav.loc[date] = value
            continue

        # Calculate daily return
        try:
            # Find the previous trading day
            current_date_loc = price_df.index.get_loc(date)
            if current_date_loc == 0:
                 # This case is handled by the initial continue in the loop
                 # But as a defensive check:
                 nav.loc[date] = value
                 continue

            prev_day_date = price_df.index[current_date_loc - 1]
            prev_prices = price_df.loc[prev_day_date]
            today_prices = price_df.loc[date]

            # Handle potential NaNs or zeros in price data before calculating returns
            # Use .reindex to align prev/today prices and ensure all assets in w_curr are considered
            assets_in_weights = list(w_curr.keys())
            aligned_prev = prev_prices.reindex(assets_in_weights)
            aligned_today = today_prices.reindex(assets_in_weights)

            # Calculate simple return (today/prev - 1)
            # Avoid division by zero or zero/nan division
            # Set return to NaN where prev_price is 0 or NaN
            ret = (aligned_today / aligned_prev) - 1

            # Replace infinite values (from division by zero) and NaNs with 0 return
            # This assumes missing price data on a day means 0 return for that day/asset,
            # which might not always be appropriate depending on data source.
            ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Calculate portfolio return
            # Ensure weights and returns align and handle potential missing assets safely
            port_ret = sum(w_curr.get(a, 0) * ret.get(a, 0) for a in assets_in_weights)
            logging.debug(f"{date.strftime('%Y-%m-%d')} - Daily Portfolio Return: {port_ret:.4f}")

            # Update portfolio value
            value *= (1 + port_ret)

        except Exception as e:
            logging.error(f"Error calculating portfolio return on {date.strftime('%Y-%m-%d')}: {e}. Previous day's value carried over.")
            # Value remains unchanged if calculation fails


        # Record NAV for the current date
        nav.loc[date] = value

    logging.info("Backtest simulation finished.")
    return nav.dropna() # Drop NaN at the beginning if any


# --- Main ---
if __name__=="__main__":
    logging.info("Starting pipeline execution.")

    # --- Load Prices ---
    prices = load_price_data(DATA_FOLDER)
    if prices.empty:
        logging.error("No price data loaded. Exiting.")
        exit() # Exit if no price data

    logging.info(f"Successfully loaded price data for {len(prices.columns)} assets from {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}.")

    # --- Backtest ---
    nav = backtest(prices, COST)

    # --- Metrics ---
    if not nav.empty and len(nav) > 1:
        logging.info("Calculating performance metrics.")
        # Ensure financial metrics calculations handle potential edge cases
        daily = nav.pct_change().dropna()

        # Annualized Return (CAGR)
        total_return = (nav.iloc[-1] / nav.iloc[0]) if nav.iloc[0] != 0 else np.nan
        years = (nav.index[-1] - nav.index[0]).days / 365.25 if len(nav) > 1 else 0
        cagr = total_return**(1/years) - 1 if years > 0 and not np.isnan(total_return) else np.nan

        # Annualized Volatility
        ann_vol = daily.std() * np.sqrt(252) if not daily.empty else np.nan

        # Sharpe Ratio (Assuming risk-free rate is 0 for simplicity)
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if not daily.empty and daily.std() != 0 else np.nan
        # Handle case where mean is 0 but std is not (Sharpe is 0)
        if sharpe is np.nan and not daily.empty and daily.std() != 0 and daily.mean() == 0:
            sharpe = 0.0

        # Maximum Drawdown
        cum = (1 + daily).cumprod() if not daily.empty else pd.Series([1])
        # Prepend 1.0 to the cumulative returns for correct drawdown calculation from start
        # Adjust index for the prepended value
        initial_index = cum.index[0] if not cum.index.empty else pd.Timestamp('today') # Use today as fallback
        prepended_index = [initial_index - pd.Timedelta(days=1)]
        cum_with_start = pd.concat([pd.Series([1.0], index=prepended_index).dropna(), cum])
        cum_with_start = cum_with_start[~cum_with_start.index.duplicated(keep='first')] # Handle duplicate index if any


        peaks = cum_with_start.cummax()
        drawdowns = (cum_with_start - peaks) / peaks
        max_dd = drawdowns.min() if not drawdowns.empty else 0


        print("\n--- Backtest Results ---")
        print(f"Period: {nav.index[0].strftime('%Y-%m-%d')} to {nav.index[-1].strftime('%Y-%m-%d')}" if not nav.empty else "Period: N/A")
        print(f"Final NAV: {nav.iloc[-1]:.4f}" if not nav.empty else "Final NAV: N/A")
        print(f"Total Return: {total_return - 1:.2%}" if not np.isnan(total_return) else "Total Return: N/A")
        print(f"CAGR: {cagr:.2%}" if not np.isnan(cagr) else "CAGR: N/A")
        print(f"Annualized Volatility: {ann_vol:.2%}" if not np.isnan(ann_vol) else "Annualized Volatility: N/A")
        print(f"Sharpe Ratio: {sharpe:.2f}" if not np.isnan(sharpe) else "Sharpe Ratio: N/A")
        print(f"Maximum Drawdown: {max_dd:.2%}" if not np.isnan(max_dd) else "Maximum Drawdown: N/A")
        print("----------------------")

        # --- Plot ---
        logging.info("Generating plot.")
        plt.figure(figsize=(12,7))
        plt.plot(nav, label="BL+Value+Growth+ERC")
        plt.title("Integrated BL Backtest NAV")
        plt.xlabel("Date")
        plt.ylabel("NAV")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        logging.warning("NAV series is empty or too short to calculate metrics/plot.")
        print("\n--- Backtest Results ---")
        print("Could not perform backtest or calculate metrics/plot due to insufficient NAV data.")
        print("----------------------")


    logging.info("Pipeline execution finished.")