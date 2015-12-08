# Determination of Star Centers based on Radon Transform (Pueyo et al., 2015).

This code is firstly used to determine the centers of the stars for the HST-STIS coronagraphic archive.

## Installation
Checkout the code from this Github repository. Then open up a terminal window and navigate to the directory for this package. Run the following command to have an installation that will evolve with the development of this codebase.
```
$ python setup.py develop
```

## Running the code:
```python
import radonCenter
(x_cen, y_cen) = radonCenter.searchCenter(image, x_ctr_assign, y_ctr_assign, size_window = image.shape[0]/2)
```

### Inputs:
1. `image`: 2d array.
2. `x_ctr_assign`: the assigned x-center, or starting x-position; for STIS, the "CRPIX1" header is suggested.
3. `y_ctr_assign`: the assigned y-center, or starting y-position; for STIS, the "CRPIX2" header is suggested.
4. `size_window`: half width of the window to generate the cost function; for STIS, half the length of the image is suggested.


## References:
Pueyo, L., Soummer, R., Hoffmann, J., et al. 2015, [ApJ, 803, 31](https://ui.adsabs.harvard.edu/#abs/2015ApJ...803...31P/abstract)
