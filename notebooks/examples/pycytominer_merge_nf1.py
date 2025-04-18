#!/usr/bin/env python

"""
Demonstrating CytoTable capabilities with input datasets.
Note: intended to be used for profiling via memray.
"""

import pathlib
import sys

from pycytominer.cyto_utils.cells import SingleCells


def main():
    input_file = sys.argv[1]
    dest_path = (
        f"{pathlib.Path(__file__).parent.resolve()}/"
        f"{pathlib.Path(input_file).name}.pycytominer.parquet"
    )

    result = SingleCells(
        sql_file=f"sqlite:///{input_file}",
        compartments=["Per_Cells", "Per_Cytoplasm", "Per_Nuclei"],
        compartment_linking_cols={
            "Per_Cytoplasm": {
                "Per_Cells": "Cytoplasm_Parent_Cells",
                "Per_Nuclei": "Cytoplasm_Parent_Nuclei",
            },
            "Per_Cells": {"Per_Cytoplasm": "Cells_Number_Object_Number"},
            "Per_Nuclei": {"Per_Cytoplasm": "Nuclei_Number_Object_Number"},
        },
        image_table_name="Per_Image",
        strata=["Image_Metadata_Well", "Image_Metadata_Plate"],
        merge_cols=["ImageNumber"],
        image_cols="ImageNumber",
        load_image_data=True,
        # perform merge_single_cells without annotation
        # and receive parquet filepath
    ).merge_single_cells(
        sc_output_file=dest_path,
        output_type="parquet",
    )

    # clean up file
    pathlib.Path(dest_path).unlink()


if __name__ == "__main__":
    main()
