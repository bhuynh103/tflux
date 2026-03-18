from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import landscape, portrait
from reportlab.pdfgen import canvas


def pngs_to_pdf(
    input_dir: str | Path,
    output_path: str | Path = "output.pdf",
    pattern: str = "*/*.png",
    sort_key=None,
    fit_to_image: bool = True,
) -> Path:
    """
    Convert all files matching pattern in a directory to a single PDF, one image per page.

    Args:
        input_dir:     Path to directory containing .png files.
        output_path:   Destination path for the generated PDF.
        pattern:       Pattern to match in directory to search for images.
                       Defaults to searching within each of input_dir's 
                       subdirectories.
        sort_key:      Optional callable used to sort the PNG files before
                       assembling pages (e.g. sort_key=lambda p: p.stem).
                       Defaults to alphabetical / natural filename order.
        fit_to_image:  If True (default), each PDF page is sized to exactly
                       match its source image's pixel dimensions (at 72 dpi).
                       If False, pages are US Letter and the image is scaled
                       to fit while preserving aspect ratio.

    Returns:
        The resolved Path of the written PDF.

    Raises:
        FileNotFoundError: If input_dir does not exist.
        ValueError:        If no PNG files are found in input_dir.
    """
    input_dir = Path(input_dir).resolve()
    output_path = Path(output_path).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    png_files = sorted(input_dir.glob(pattern=pattern), key=sort_key)
    if not png_files:
        raise ValueError(f"No PNG files found in: {input_dir}")

    # Use the first image to initialise the canvas page size
    first_img = Image.open(png_files[0])
    if fit_to_image:
        init_pagesize = (first_img.width, first_img.height)
    else:
        init_pagesize = portrait((612, 792))  # US Letter in points

    c = canvas.Canvas(str(output_path), pagesize=init_pagesize)

    for png_path in png_files:
        with Image.open(png_path) as img:
            img_w, img_h = img.size  # pixels → treated as points at 72 dpi

        if fit_to_image:
            page_w, page_h = img_w, img_h
        else:
            page_w, page_h = (
                landscape((612, 792)) if img_w > img_h else portrait((612, 792))
            )
            # Scale image to fill the page while preserving aspect ratio
            scale = min(page_w / img_w, page_h / img_h)
            img_w, img_h = img_w * scale, img_h * scale

        c.setPageSize((page_w, page_h))

        # Centre the image on the page
        x = (page_w - img_w) / 2
        y = (page_h - img_h) / 2

        c.drawImage(str(png_path), x, y, width=img_w, height=img_h)
        c.showPage()

    c.save()
    print(f"PDF written to: {output_path}  ({len(png_files)} pages)")
    return output_path