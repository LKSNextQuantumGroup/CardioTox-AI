# convert_visits_to_parquet.py

from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).parent

    input_path = base_dir / "data" / "visits_dataton_actualizado.csv"
    output_path = base_dir / "data" / "visits_dataton_actualizado.parquet"

    print("=" * 80)
    print("CONVERSIÓN CSV → PARQUET")
    print("=" * 80)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el fichero:\n{input_path}"
        )

    print(f"\nLeyendo CSV:\n{input_path}")

    df = pd.read_csv(input_path)

    print(f"\nFilas: {len(df):,}")
    print(f"Columnas: {len(df.columns)}")

    print(f"\nGuardando parquet:\n{output_path}")

    df.to_parquet(
        output_path,
        index=False,
        compression="snappy",
    )

    csv_size_mb = input_path.stat().st_size / (1024 * 1024)
    parquet_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 80)
    print("CONVERSIÓN FINALIZADA")
    print("=" * 80)

    print(f"\nTamaño CSV:     {csv_size_mb:.2f} MB")
    print(f"Tamaño Parquet: {parquet_size_mb:.2f} MB")

    reduction = 100 * (1 - parquet_size_mb / csv_size_mb)

    print(f"Reducción:      {reduction:.1f}%")

    print("\nFichero generado:")
    print(output_path)


if __name__ == "__main__":
    main()