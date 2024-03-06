from pretty_html_table import build_table
from pathlib import Path, PurePath
import pandas as pd


def make_html_table(df: pd.DataFrame) -> str:
    html_table = build_table(df, 'blue_light', padding='10px')
    return html_table


def make_and_save_html_table(df: pd.DataFrame, output: Path) -> None:
    html_table = make_html_table(df)
    output.write_text(html_table)
