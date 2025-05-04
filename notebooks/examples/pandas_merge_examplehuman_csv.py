#!/usr/bin/env python

"""
Demonstrating CytoTable capabilities with input datasets using pandas only.
Note: intended to be used for profiling via memray.
"""

import pathlib
import sys

import pandas as pd


def col_renames(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to rename columns appropriately
    """

    def rename(colname):
        if colname in ["ImageNumber", "ObjectNumber"]:
            return f"Metadata_{colname}"
        elif any(parent in colname for parent in ["Parent_Cells", "Parent_Nuclei"]):
            return f"Metadata_{name}_{colname}"
        elif not (colname.startswith(name) or colname.startswith("Metadata_")):
            return f"{name}_{colname}"
        return colname

    return df.rename(columns={col: rename(col) for col in df.columns})


def main():
    input_file = sys.argv[1]
    dest_path = (
        f"{pathlib.Path(__file__).parent.resolve()}/"
        f"{pathlib.Path(input_file).name}.pycytominer.parquet"
    )

    image_df = pd.read_csv(
        f"{input_file}/Image.csv",
        usecols=[
            "ImageNumber",
            "FileName_DNA",
            "FileName_OrigOverlay",
            "FileName_PH3",
            "FileName_cellbody",
        ],
    )
    cytoplasm_df = pd.read_csv(f"{input_file}/Cytoplasm.csv")
    cells_df = pd.read_csv(f"{input_file}/Cells.csv")
    nuclei_df = pd.read_csv(f"{input_file}/Nuclei.csv")

    image_df = col_renames("Image", image_df)
    cytoplasm_df = col_renames("Cytoplasm", cytoplasm_df)
    cells_df = col_renames("Cells", cells_df)
    nuclei_df = col_renames("Nuclei", nuclei_df)

    # Merge operations
    df = (
        image_df.merge(
            cytoplasm_df,
            how="left",
            left_on="Metadata_ImageNumber",
            right_on="Metadata_ImageNumber",
        )
        .merge(
            cells_df,
            how="left",
            left_on=["Metadata_ImageNumber", "Metadata_Cytoplasm_Parent_Cells"],
            right_on=["Metadata_ImageNumber", "Metadata_ObjectNumber"],
            suffixes=("", "_Cells"),
        )
        .merge(
            nuclei_df,
            how="left",
            left_on=["Metadata_ImageNumber", "Metadata_Cytoplasm_Parent_Nuclei"],
            right_on=["Metadata_ImageNumber", "Metadata_ObjectNumber"],
            suffixes=("", "_Nuclei"),
        )
    )

    # Drop duplicate columns caused by multiple ObjectNumber columns
    df = df.loc[:, ~df.columns.duplicated()]

    # Clean up output path if it exists (optional — doesn't actually save in this example)
    output_path = pathlib.Path(dest_path)
    if output_path.exists():
        output_path.unlink()

    # Save the merged DataFrame to a parquet file
    df.to_parquet(dest_path, index=False)

    # clean up file
    pathlib.Path(dest_path).unlink()


if __name__ == "__main__":
    main()
