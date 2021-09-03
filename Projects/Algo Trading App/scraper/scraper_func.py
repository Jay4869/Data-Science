from datetime import datetime
from lxml import html
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from itertools import compress


def get_page(url):
    # Set up the request headers that we're going to use, to simulate
    # a request by the Chrome browser. Simulating a request from a browser
    # is generally good practice when building a scraper
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "max-age=0",
        "Pragma": "no-cache",
        "Referrer": "https://google.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36",
    }
    return requests.get(url, headers=headers)


def parse_rows(table_rows):
    parsed_rows = []

    for table_row in table_rows:
        parsed_row = []
        el = table_row.xpath("./div")

        none_count = 0

        for rs in el:
            try:
                (text,) = rs.xpath(".//span/text()[1]")
                parsed_row.append(text)
            except ValueError:
                parsed_row.append(np.NaN)
                none_count += 1

        if none_count < 4:
            parsed_rows.append(parsed_row)

    return pd.DataFrame(parsed_rows)


def clean_data(df):
    df = df.set_index(0)  # Set the index to the first column: 'Period Ending'.
    df = (
        df.transpose()
    )  # Transpose the DataFrame, so that our header contains the account names

    # Rename the "Breakdown" column to "Date"
    cols = list(df.columns)
    cols[0] = "Date"
    df = df.set_axis(cols, axis="columns", inplace=False)

    numeric_columns = list(df.columns)[
        1::
    ]  # Take all columns, except the first (which is the 'Date' column)

    for column_index in range(
        1, len(df.columns)
    ):  # Take all columns, except the first (which is the 'Date' column)
        df.iloc[:, column_index] = df.iloc[:, column_index].str.replace(
            ",", ""
        )  # Remove the thousands separator
        df.iloc[:, column_index] = df.iloc[:, column_index].astype(
            np.float64
        )  # Convert the column to float64

    return df


def scrape_table(url, test=False):
    # Fetch the page that we're going to parse
    # rest_interval=6*random.random()
    # time.sleep(rest_interval)
    try:
        page = get_page(url)

        # Parse the page with LXML, so that we can start doing some XPATH queries
        # to extract the data that we want
        tree = html.fromstring(page.content)

        # Fetch all div elements which have class 'D(tbr)'
        table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")

        # Ensure that some table rows are found; if none are found, then it's possible
        # that Yahoo Finance has changed their page layout, or have detected
        # that you're scraping the page.
        assert len(table_rows) > 0
    except:
        return False
    if test == True:
        return True
    else:
        df = parse_rows(table_rows)
        df = clean_data(df)
        return df


def scrape_key_stat(url):
    # rest_interval=6*random.random()
    # time.sleep(rest_interval)
    temp_dir = {}
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    tabl = soup.findAll(
        "table", {"class": "W(100%) Bdcl(c)"}
    )  # try soup.findAll("table") if this line gives error
    for t in tabl:
        rows = t.find_all("tr")
        for row in rows:
            if len(row.get_text(separator="|").split("|")[0:2]) > 0:
                temp_dir[row.get_text(separator="|").split("|")[0]] = row.get_text(
                    separator="|"
                ).split("|")[-1]
    df = pd.DataFrame.from_dict(temp_dir, orient="index")
    df = df.replace({",": ""}, regex=True)
    df = df.replace({"M": "E+03"}, regex=True)
    df = df.replace({"B": "E+06"}, regex=True)
    df = df.replace({"T": "E+09"}, regex=True)
    df = df.replace({"%": "E-02"}, regex=True)
    df = df.apply(pd.to_numeric, errors="coerce").transpose()
    df.columns = df.columns.str.lower()
    df = df.filter(
        ["market cap (intraday)", "enterprise value", "forward annual dividend yield"]
    )
    return df


def scrape(symbol):
    print("Attempting to scrape data for " + symbol)
    df_balance_sheet = scrape_table(
        "https://ca.finance.yahoo.com/quote/" + symbol + "/balance-sheet?p=" + symbol
    )
    Date = df_balance_sheet["Date"][1]
    df_balance_sheet = df_balance_sheet.set_index("Date")

    df_income_statement = scrape_table(
        "https://ca.finance.yahoo.com/quote/" + symbol + "/financials?p=" + symbol
    )
    df_income_statement = df_income_statement.set_index("Date")

    df_cash_flow = scrape_table(
        "https://ca.finance.yahoo.com/quote/" + symbol + "/cash-flow?p=" + symbol
    )
    df_cash_flow = df_cash_flow.set_index("Date")

    df_key_stat = scrape_key_stat(
        "https://ca.finance.yahoo.com/quote/" + symbol + "/key-statistics?p=" + symbol
    )
    df_key_stat["Date"] = Date
    df_key_stat = df_key_stat.set_index("Date")

    df_joined = (
        df_balance_sheet.join(
            df_income_statement, on="Date", how="outer", rsuffix=" - Income Statement"
        )
        .join(df_cash_flow, on="Date", how="outer", rsuffix=" - Cash Flow")
        .join(df_key_stat, on="Date", how="outer", rsuffix=" - Key Stat")
        .dropna(axis=1, how="all")
    )
    df_joined = df_joined.loc[:, ~df_joined.columns.duplicated()].reset_index().head(1)
    df_joined.columns = df_joined.columns.str.lower()
    df_joined.insert(1, "symbol", symbol)
    df_joined.drop("index", 1)
    magic = [
        "Symbol",
        "Date",
        "Enterprise Value",
        "EBITDA",
        "Depreciation & amortization",
        "Market Cap (intraday)",
        "Net income available to common shareholders",
        "net cash provided by operating activites",
        "Capital Expenditure",
        "Total Current Assets",
        "Total Current Liabilities",
        "Net property, plant and equipment",
        "Total stockholders' equity",
        "Long Term Debt",
        "Forward Annual Dividend Yield",
        "Total assets",
        "Other long-term liabilities",
        "Common stock",
        "Total revenue",
        "Gross profit",
    ]

    return df_joined.filter([x.lower() for x in magic]).infer_objects()


def check_valid(symbols):
    temp = [
        scrape_table(
            "https://ca.finance.yahoo.com/quote/"
            + symbol
            + "/balance-sheet?p="
            + symbol,
            test=True,
        )
        for symbol in symbols
    ]
    return list(compress(symbols, temp))


def scrape_multi(symbols):
    symbols = check_valid(symbols)
    return pd.concat([scrape(symbol) for symbol in symbols], sort=False)


def top_value_stock(df, method="combined"):
    df.columns = [
        "Symbol",
        "Date",
        "EV",
        "EBITDA",
        "D&A",
        "MarketCap",
        "NetIncome",
        "CashFlowOps",
        "Capex",
        "CurrAsset",
        "CurrLiab",
        "PPE",
        "BookValue",
        "LTDebt",
        "DivYield",
        "TotAssets",
        "OtherLTDebt",
        "CommStock",
        "TotRevenue",
        "GrossProfit",
    ]
    df["EBIT"] = df["EBITDA"] - df["D&A"]
    df["EarningYield"] = df["EBIT"] / df["EV"]
    df["FCFYield"] = (df["CashFlowOps"] - df["Capex"]) / df["MarketCap"]
    df["ROC"] = df["EBIT"] / (df["PPE"] + df["CurrAsset"] - df["CurrLiab"])
    df["BookToMkt"] = df["BookValue"] / df["MarketCap"]

    if method == "magic":
        # finding value stocks based on Magic Formula
        df = df.loc[df["ROC"] > 0]
        df["CombRank"] = df["EarningYield"].rank(
            ascending=False, na_option="bottom"
        ) + df["ROC"].rank(ascending=False, na_option="bottom")
        df["MagicFormulaRank"] = df["CombRank"].rank(method="first")
        value_stocks = df.sort_values("MagicFormulaRank")[
            ["Symbol", "MagicFormulaRank", "EarningYield", "ROC"]
        ]
        return value_stocks
    if method == "dividend":
        # finding highest dividend yield stocks
        df = df.loc[df["ROC"] > 0]
        high_dividend_stocks = df.sort_values("DivYield", ascending=False)[
            ["Symbol", "MagicFormulaRank", "DivYield", "ROC"]
        ]
        return high_dividend_stocks
    if method == "piotroski_f":
        df = piotroski_f(df)
        high_f_stocks = df.sort_values("Piotroski_f", ascending=False)[
            ["Symbol", "Piotroski_f", "DivYield", "ROC"]
        ]
        return high_f_stocks
    else:
        # # Magic Formula & Dividend yield combined
        df = df.loc[df["ROC"] > 0]
        df["CombRank"] = (
            df["EarningYield"].rank(ascending=False, method="first")
            + df["ROC"].rank(ascending=False, method="first")
            + df["DivYield"].rank(ascending=False, method="first")
        )
        df["CombinedRank"] = df["CombRank"].rank(method="first")
        value_high_div_stocks = df.sort_values("CombinedRank")[
            ["Symbol", "EarningYield", "DivYield", "ROC", "CombinedRank"]
        ]
        return value_high_div_stocks


def piotroski_f(df):
    """function to calculate f score of each stock and output information as dataframe"""
    df["ROA_FS"] = (
        df["NetIncome"] / ((df["TotAssets"] + df["TotAssets"]) / 2) > 0
    ).astype(int)
    df["CFO_FS"] = (df["CashFlowOps"] > 0).astype(int)
    df["ROA_D_FS"] = (
        df["NetIncome"] / (df["TotAssets"] + df["TotAssets"]) / 2
        > df["NetIncome"] / (df["TotAssets"] + df["TotAssets"]) / 2
    ).astype(int)
    df["CFO_ROA_FS"] = (
        df["CashFlowOps"] / df["TotAssets"]
        > df["NetIncome"] / ((df["TotAssets"] + df["TotAssets"]) / 2)
    ).astype(int)
    df["LTD_FS"] = (
        (df["LTDebt"] + df["OtherLTDebt"]) < (df["LTDebt"] + df["OtherLTDebt"])
    ).astype(int)
    df["CR_FS"] = (
        (df["CurrAsset"] / df["CurrLiab"]) > (df["CurrAsset"] / df["CurrLiab"])
    ).astype(int)
    df["DILUTION_FS"] = (df["CommStock"] <= df["CommStock"]).astype(int)
    df["GM_FS"] = (
        (df["GrossProfit"] / df["TotRevenue"]) > (df["GrossProfit"] / df["TotRevenue"])
    ).astype(int)
    df["ATO_FS"] = (
        df["TotRevenue"] / ((df["TotAssets"] + df["TotAssets"]) / 2)
        > df["TotRevenue"] / ((df["TotAssets"] + df["TotAssets"]) / 2)
    ).astype(int)
    df["Piotroski_f"] = (
        df["ROA_FS"]
        + df["CFO_FS"]
        + df["ROA_D_FS"]
        + df["CFO_ROA_FS"]
        + df["LTD_FS"]
        + df["CR_FS"]
        + df["DILUTION_FS"]
        + df["GM_FS"]
        + df["ATO_FS"]
    )
    return df.drop(
        [
            "ROA_FS",
            "CFO_FS",
            "ROA_D_FS",
            "CFO_ROA_FS",
            "LTD_FS",
            "CR_FS",
            "DILUTION_FS",
            "GM_FS",
            "ATO_FS",
        ],
        axis=1,
    )


def scrp_unit_test():
    tickers = [
        "UTX",
        "UNH",
        "VZ",
        "V",
        "AXP",
        "AAPL",
        "BA",
        "CAT",
        "CVX",
        "CSCO",
    ]

    d = scrape_multi(tickers)
    st = top_value_stock(d, method="piotroski_f")
    return st