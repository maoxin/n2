from pathlib import Path
if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import concurrent.futures

import gdelt
from tqdm import tqdm

from data.utils import get_date_range, to_datestr


def download_(date, output_dir=None):
    if output_dir is None:
        output_dir = Path.cwd().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    gd2 = gdelt.gdelt(version=2)
    csv = gd2.Search(date=to_datestr(date, format="%Y %m %d"), table="events", coverage=True, translation=False, output="csv")
    save_name = f"{to_datestr(date, '%Y%m%d')}.csv"
    with open(output_dir/save_name, "w") as f:
        f.write(csv)


def download(start_date, end_date=None, output_dir=None, multiprocessing=False):
    if end_date is None:
        end_date = start_date
    date_range = get_date_range(start_date, end_date)
    if multiprocessing:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in tqdm(executor.map(download_, date_range, output_dir=output_dir), total=len(date_range), desc="Downloading GDELT records"):
                pass
    else:
        for date_str in tqdm(date_range, desc="Downloading GDELT records"):
            download_(date_str, output_dir=output_dir)


if __name__ == "__main__":
    start_date = "20220726"
    end_date = "20220727"
    download(start_date, end_date, output_dir="/Volumes/Extreme SSD/gdelt_archive", multiprocessing=False)
