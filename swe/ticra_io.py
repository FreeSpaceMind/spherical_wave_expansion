"""
Ticra GRASP File I/O Module

Readers and writers for Ticra GRASP file formats:
- .sph: Spherical wave expansion coefficients
- .cut: Far-field pattern cuts (spherical cuts)
- .grd: Field grid data (planar near-field, far-field grids)

File format references:
- GRASP Technical Description (TICRA)
- python-graspfile (Smithsonian/python-graspfile)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# .cut File I/O (GRASP spherical cut format)
# ==============================================================================

def read_grasp_cut(filename: str) -> Dict:
    """
    Read a GRASP .cut file containing far-field (or near-field) pattern cuts.

    The .cut format stores field data along angular cuts. For spherical cuts
    with ICUT=1, each cut is at a fixed phi with theta as the swept variable.

    Args:
        filename: Path to the .cut file

    Returns:
        Dictionary containing:
            - 'cuts': list of dicts, each with keys:
                - 'v_ini': initial angle (degrees)
                - 'v_inc': angle increment (degrees)
                - 'v_num': number of points
                - 'constant': constant coordinate value (degrees)
                - 'icomp': polarization definition (1=Etheta/Ephi, 2=RHCP/LHCP, 3=Eco/Ecx)
                - 'icut': cut type (1=fixed phi, 2=fixed theta)
                - 'ncomp': number of field components (2=far, 3=near)
                - 'data': complex array shape (v_num, ncomp)
    """
    logger.info(f"Reading GRASP .cut file: {filename}")

    cuts = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        parts = line.split()

        # A spec line has exactly 7 numeric values
        if len(parts) == 7:
            try:
                v_ini = float(parts[0])
                v_inc = float(parts[1])
                v_num = int(parts[2])
                constant = float(parts[3])
                icomp = int(parts[4])
                icut = int(parts[5])
                ncomp = int(parts[6])
            except (ValueError, IndexError):
                i += 1
                continue

            i += 1

            # Read data lines
            data = np.zeros((v_num, ncomp), dtype=complex)
            for j in range(v_num):
                if i >= len(lines):
                    break
                dline = lines[i].strip()
                if not dline:
                    i += 1
                    j -= 1  # retry
                    continue
                dparts = dline.split()
                # Skip stray "Field" comment lines
                if dparts[0] == "Field":
                    i += 1
                    continue
                data[j, 0] = complex(float(dparts[0]), float(dparts[1]))
                data[j, 1] = complex(float(dparts[2]), float(dparts[3]))
                if ncomp == 3 and len(dparts) >= 6:
                    data[j, 2] = complex(float(dparts[4]), float(dparts[5]))
                i += 1

            cut = {
                'v_ini': v_ini,
                'v_inc': v_inc,
                'v_num': v_num,
                'constant': constant,
                'icomp': icomp,
                'icut': icut,
                'ncomp': ncomp,
                'data': data,
            }
            cuts.append(cut)
            logger.debug(
                f"Read cut: constant={constant}, v_ini={v_ini}, v_inc={v_inc}, "
                f"v_num={v_num}, icomp={icomp}, icut={icut}, ncomp={ncomp}"
            )
        else:
            # Text/comment line - skip
            i += 1

    logger.info(f"Read {len(cuts)} cuts from {filename}")
    return {'cuts': cuts}


def write_grasp_cut(filename: str, cuts: List[Dict],
                    text_header: str = "SWE generated cut file"):
    """
    Write a GRASP .cut file.

    Args:
        filename: Output file path
        cuts: List of cut dictionaries (same format as read_grasp_cut output)
        text_header: Text description line written before each cut
    """
    logger.info(f"Writing GRASP .cut file: {filename}")

    with open(filename, 'w') as f:
        for cut in cuts:
            # Text line
            f.write(f"{text_header}\n")

            # Spec line
            f.write(f" {cut['v_ini']:.10E} {cut['v_inc']:.10E} "
                    f"{cut['v_num']:d} {cut['constant']:.10E} "
                    f"{cut['icomp']:d} {cut['icut']:d} {cut['ncomp']:d}\n")

            # Data lines
            data = cut['data']
            ncomp = cut['ncomp']
            for j in range(cut['v_num']):
                line = (f" {data[j, 0].real:20.10E} {data[j, 0].imag:20.10E}"
                        f" {data[j, 1].real:20.10E} {data[j, 1].imag:20.10E}")
                if ncomp == 3:
                    line += f" {data[j, 2].real:20.10E} {data[j, 2].imag:20.10E}"
                f.write(line + "\n")

    logger.info(f"Wrote {len(cuts)} cuts to {filename}")


def cut_to_fields(cut_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract theta, phi, E_theta, E_phi arrays from cut file data.

    Assumes ICUT=1 (standard polar cuts with fixed phi, varying theta)
    and ICOMP=1 (E_theta/E_phi components).

    Args:
        cut_data: Dictionary returned by read_grasp_cut

    Returns:
        theta: 1D array of theta values (radians) for each point in all cuts
        phi: 1D array of phi values (radians) for each point in all cuts
        E_theta: 1D complex array
        E_phi: 1D complex array
    """
    theta_list = []
    phi_list = []
    E_theta_list = []
    E_phi_list = []

    for cut in cut_data['cuts']:
        if cut['icut'] != 1:
            logger.warning(f"Skipping cut with icut={cut['icut']} (only icut=1 supported)")
            continue

        angles_deg = cut['v_ini'] + np.arange(cut['v_num']) * cut['v_inc']
        theta_rad = np.deg2rad(angles_deg)
        phi_rad = np.deg2rad(cut['constant']) * np.ones(cut['v_num'])

        theta_list.append(theta_rad)
        phi_list.append(phi_rad)

        if cut['icomp'] == 1:
            # Already E_theta / E_phi
            E_theta_list.append(cut['data'][:, 0])
            E_phi_list.append(cut['data'][:, 1])
        else:
            logger.warning(f"icomp={cut['icomp']} not yet supported for conversion, "
                           "treating as E_theta/E_phi")
            E_theta_list.append(cut['data'][:, 0])
            E_phi_list.append(cut['data'][:, 1])

    theta = np.concatenate(theta_list)
    phi = np.concatenate(phi_list)
    E_theta = np.concatenate(E_theta_list)
    E_phi = np.concatenate(E_phi_list)

    return theta, phi, E_theta, E_phi


# ==============================================================================
# .grd File I/O (GRASP grid format)
# ==============================================================================

def read_grasp_grd(filename: str) -> Dict:
    """
    Read a GRASP .grd file containing field data on a rectangular grid.

    Args:
        filename: Path to the .grd file

    Returns:
        Dictionary containing:
            - 'header': list of header text lines
            - 'ktype': file type (1=standard)
            - 'nset': number of field sets
            - 'icomp': polarization type (same codes as .cut ICOMP)
            - 'ncomp': number of field components (2=far, 3=near)
            - 'igrid': grid type (1=uv, 7=theta-phi, etc.)
            - 'beam_centers': list of (ix, iy) tuples
            - 'fields': list of dicts, each with:
                - 'grid_min_x', 'grid_min_y', 'grid_max_x', 'grid_max_y': grid extents
                - 'nx', 'ny': number of grid points
                - 'klimit': 0=filled, 1=sparse
                - 'data': complex array shape (ny, nx, ncomp)
    """
    logger.info(f"Reading GRASP .grd file: {filename}")

    with open(filename, 'r') as f:
        # Read header lines until "++++
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file before '++++'")
            if line.strip().startswith("++++"):
                break
            header.append(line.rstrip('\n'))

        # KTYPE
        ktype = int(f.readline().strip())

        # NSET, ICOMP, NCOMP, IGRID
        parts = f.readline().strip().split()
        nset = int(parts[0])
        icomp = int(parts[1])
        ncomp = int(parts[2])
        igrid = int(parts[3])

        # Beam centers
        beam_centers = []
        for _ in range(nset):
            parts = f.readline().strip().split()
            beam_centers.append((int(parts[0]), int(parts[1])))

        # Read field sets
        fields = []
        for s in range(nset):
            # Grid extents
            parts = f.readline().strip().split()
            grid_min_x = float(parts[0])
            grid_min_y = float(parts[1])
            grid_max_x = float(parts[2])
            grid_max_y = float(parts[3])

            # Grid size
            parts = f.readline().strip().split()
            nx = int(parts[0])
            ny = int(parts[1])
            klimit = int(parts[2])

            # Read data
            data = np.zeros((ny, nx, ncomp), dtype=complex)
            for j in range(ny):
                if klimit == 1:
                    parts = f.readline().strip().split()
                    i_start = int(parts[0]) - 1
                    i_count = int(parts[1])
                else:
                    i_start = 0
                    i_count = nx

                for i in range(i_start, i_start + i_count):
                    parts = f.readline().strip().split()
                    data[j, i, 0] = complex(float(parts[0]), float(parts[1]))
                    data[j, i, 1] = complex(float(parts[2]), float(parts[3]))
                    if ncomp == 3 and len(parts) >= 6:
                        data[j, i, 2] = complex(float(parts[4]), float(parts[5]))

            field = {
                'grid_min_x': grid_min_x,
                'grid_min_y': grid_min_y,
                'grid_max_x': grid_max_x,
                'grid_max_y': grid_max_y,
                'nx': nx,
                'ny': ny,
                'klimit': klimit,
                'data': data,
            }
            fields.append(field)

            logger.debug(
                f"Read field set {s}: grid=[{grid_min_x},{grid_min_y}]->[{grid_max_x},{grid_max_y}], "
                f"size={nx}x{ny}, klimit={klimit}"
            )

    result = {
        'header': header,
        'ktype': ktype,
        'nset': nset,
        'icomp': icomp,
        'ncomp': ncomp,
        'igrid': igrid,
        'beam_centers': beam_centers,
        'fields': fields,
    }

    logger.info(f"Read {nset} field sets from {filename}")
    return result


def write_grasp_grd(filename: str, grd_data: Dict):
    """
    Write a GRASP .grd file.

    Args:
        filename: Output file path
        grd_data: Dictionary in the same format as read_grasp_grd output
    """
    logger.info(f"Writing GRASP .grd file: {filename}")

    with open(filename, 'w') as f:
        # Header
        for line in grd_data.get('header', ['SWE generated grid file']):
            f.write(line + "\n")
        f.write("++++\n")

        # KTYPE
        f.write(f"{grd_data['ktype']:d}\n")

        # NSET, ICOMP, NCOMP, IGRID
        f.write(f"{grd_data['nset']:d} {grd_data['icomp']:d} "
                f"{grd_data['ncomp']:d} {grd_data['igrid']:d}\n")

        # Beam centers
        for bc in grd_data['beam_centers']:
            f.write(f"{bc[0]:d} {bc[1]:d}\n")

        # Field sets
        for field in grd_data['fields']:
            f.write(f"{field['grid_min_x']:.10E} {field['grid_min_y']:.10E} "
                    f"{field['grid_max_x']:.10E} {field['grid_max_y']:.10E}\n")
            f.write(f"{field['nx']:d} {field['ny']:d} 0\n")

            data = field['data']
            ncomp = grd_data['ncomp']
            for j in range(field['ny']):
                for i in range(field['nx']):
                    line = (f"{data[j, i, 0].real:.10E} {data[j, i, 0].imag:.10E} "
                            f"{data[j, i, 1].real:.10E} {data[j, i, 1].imag:.10E}")
                    if ncomp == 3:
                        line += f" {data[j, i, 2].real:.10E} {data[j, i, 2].imag:.10E}"
                    f.write(line + "\n")

    logger.info(f"Wrote {len(grd_data['fields'])} field sets to {filename}")
