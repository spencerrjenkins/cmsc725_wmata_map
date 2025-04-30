import sys
import geopandas as gpd
from pyproj import CRS


def convert_geojson_crs(input_path, from_epsg, to_epsg):
    gdf = gpd.read_file(input_path)
    gdf = gdf.set_crs(epsg=int(from_epsg), allow_override=True)
    gdf = gdf.to_crs(epsg=int(to_epsg))
    output_path = input_path.replace(".geojson", f"_crs{to_epsg}.geojson")
    gdf.to_file(output_path, driver="GeoJSON")
    print(
        f"Converted {input_path} from EPSG:{from_epsg} to EPSG:{to_epsg} and saved as {output_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python geojson_converter.py <path-to-file> <from-epsg> <to-epsg>")
        sys.exit(1)
    input_path, from_epsg, to_epsg = sys.argv[1:4]
    convert_geojson_crs(input_path, from_epsg, to_epsg)
