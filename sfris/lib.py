"""
sfris.lib - Starting structure library (convenience feature).

Pre-computed GFN2-xTB geometries for molecular formulas covered by the curated
QM9 dataset. These structures are NOT validated minima, have NOT been refined
at any higher level, and are NOT a substitute for a SFRIS run. They exist only
to spare the user the effort of preparing an input geometry by hand.

Coverage: C/H/O/N/F, up to 9 heavy atoms.

All coordinates are GFN2-xTB minima computed with SFRIS. No geometry from the
source dataset is reproduced here; source geometries were used only as starting
points and were replaced in the first optimization step.

Data provenance:
    curated QM9   Senthil, Chakraborty, Ramakrishnan, Chem Sci 2021, 12, 5566
    QM9           Ramakrishnan, Dral, Rupp, von Lilienfeld, Sci Data 2014, 1, 140022
    GDB-17        Ruddigkeit, van Deursen, Blum, Reymond, JCIM 2012, 52, 2864

Any use of this library should cite all three in addition to SFRIS.
"""

import gzip
import json
import os
import re

_DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'startlib.json.gz')
_LIB = None

BIBTEX = """@article{Ramakrishnan2014,
  author  = {Ramakrishnan, R. and Dral, P. O. and Rupp, M. and von Lilienfeld, O. A.},
  title   = {Quantum chemistry structures and properties of 134 kilo molecules},
  journal = {Scientific Data},
  volume  = {1},
  pages   = {140022},
  year    = {2014},
  doi     = {10.1038/sdata.2014.22}
}

@article{Ruddigkeit2012,
  author  = {Ruddigkeit, L. and van Deursen, R. and Blum, L. C. and Reymond, J.-L.},
  title   = {Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {52},
  number  = {11},
  pages   = {2864--2875},
  year    = {2012},
  doi     = {10.1021/ci300415d}
}

@article{Senthil2021,
  author  = {Senthil, S. and Chakraborty, S. and Ramakrishnan, R.},
  title   = {Troubleshooting unstable molecules in chemical space},
  journal = {Chemical Science},
  volume  = {12},
  number  = {16},
  pages   = {5566--5573},
  year    = {2021},
  doi     = {10.1039/D0SC05591C}
}
"""

class LibraryError(Exception):
    """Raised for formula parsing failures and missing library entries."""
    pass

def _load():
    """Load the library on first use."""
    global _LIB
    if _LIB is None:
        if not os.path.isfile(_DATA_FILE):
            raise LibraryError(
                'starting structure library not found: %s' % _DATA_FILE)
        with gzip.open(_DATA_FILE, 'rt', encoding='ascii') as fh:
            _LIB = json.load(fh)
    return _LIB

def parse_formula(text):
    """Parse a formula string into an element -> count dict.

    Case sensitive: CO is carbon monoxide, Co is cobalt.
    """
    text = text.strip()
    if not text:
        raise LibraryError('empty formula')
    counts = {}
    pos = 0
    while pos < len(text):
        m = re.match(r'([A-Z][a-z]?)(\d*)', text[pos:])
        if not m or not m.group(1):
            raise LibraryError(
                'cannot parse formula %r at position %d' % (text, pos))
        sym = m.group(1)
        num = int(m.group(2)) if m.group(2) else 1
        counts[sym] = counts.get(sym, 0) + num
        pos += m.end()
    return counts

def to_hill(counts):
    """Convert an element -> count dict to Hill notation."""
    order = []
    if 'C' in counts:
        order.append('C')
        if 'H' in counts:
            order.append('H')
        order += sorted(k for k in counts if k not in ('C', 'H'))
    else:
        order += sorted(counts)
    out = ''
    for sym in order:
        n = counts[sym]
        out += sym + (str(n) if n > 1 else '')
    return out

def normalize(text):
    """Normalize any element ordering to the Hill notation key.

    Returns (hill, was_reordered).
    """
    counts = parse_formula(text)
    lib = _load()
    unsupported = [e for e in counts if e not in lib['meta']['elements']]
    if unsupported:
        raise LibraryError(
            "SFRIS-LIB: %s is not covered by the library "
            "(library contains %s only).\n"
            "           SFRIS itself supports other elements; prepare an "
            "input geometry\n"
            "           and run sfris directly."
            % (', '.join(sorted(unsupported)),
               '/'.join(lib['meta']['elements'])))
    hill = to_hill(counts)
    return hill, (hill != text.strip())

def suggest(text, limit=5):
    """Suggest nearby formula keys for a miss. Cheap heuristic."""
    lib = _load()
    try:
        counts = parse_formula(text)
    except LibraryError:
        return []
    target_set = frozenset(counts)
    target_n = sum(counts.values())

    scored = []
    for key in lib['formulas']:
        try:
            c = parse_formula(key)
        except LibraryError:
            continue
        set_penalty = len(target_set.symmetric_difference(frozenset(c))) * 10
        n_penalty = abs(sum(c.values()) - target_n)
        scored.append((set_penalty + n_penalty, key))
    scored.sort()
    return [k for _, k in scored[:limit]]

def info():
    """Return the library metadata dict."""
    return dict(_load()['meta'])

def has(formula):
    """Return True if the formula is covered."""
    try:
        hill, _ = normalize(formula)
    except LibraryError:
        return False
    return hill in _load()['formulas']

def entry(formula):
    """Return the full library record for a formula."""
    hill, _ = normalize(formula)
    rec = _load()['formulas'].get(hill)
    if rec is None:
        raise LibraryError('%s not in library' % hill)
    return rec

def get(formula, top=1):
    """Return up to `top` starting structures for a formula."""
    return entry(formula)['structures'][:max(1, int(top))]

def to_xyz(hill, struct):
    """Render one library structure as an XYZ block.

    The comment line carries the caveat so that it travels with the file.
    """
    meta = _load()['meta']
    comment = (
        'SFRIS-LIB %s rank=%d dE=%.2f kcal/mol | %s unrefined | '
        'starting geometry only | src=curatedQM9 | lib=%s'
        % (hill, struct['rank'], struct['de'], meta['method'],
           meta['library_version'])
    )
    lines = [str(len(struct['symbols'])), comment]
    for sym, (x, y, z) in zip(struct['symbols'], struct['coords']):
        lines.append('%-2s %15.8f %15.8f %15.8f' % (sym, x, y, z))
    return '\n'.join(lines) + '\n'

def warning_text():
    """The disclaimer printed on every library access."""
    meta = _load()['meta']
    return (
        'SFRIS-LIB  WARNING\n'
        '  Convenience feature. Structures are %s level, not refined and not\n'
        '  validated. Intended as starting geometries only. Verify before use.\n'
        '  Library %s | SFRIS %s | xTB %s | CutControl = %s\n'
        '  Derived from: %s\n'
        '  Cite Ramakrishnan (2014), Ruddigkeit (2012), Senthil (2021)\n'
        '  in addition to SFRIS. See SFRIS.bib\n'
        % (meta['method'], meta['library_version'], meta['sfris_version'],
           meta['xtb_version'], meta['cut_control'], meta['source'])
    )

def write_bibtex(path):
    """Write the three library citations to a BibTeX file."""
    with open(path, 'w') as fh:
        fh.write('% Citations for the SFRIS starting structure library\n')
        fh.write(BIBTEX)
    return path

# ================================================================
# CLI - hand-rolled dispatch, matching the style of sfris.cli
# ================================================================

USAGE = """Usage:
  sfris lib info                # Library version and coverage
  sfris lib list <formula>      # List stored structures for a formula
  sfris lib get <formula>       # Write the lowest structure as xyz
  sfris lib get <formula> -n N  # Write the N lowest structures
"""

def _resolve(formula):
    """Shared lookup: Hill normalization, then suggestions on a miss."""
    hill, reordered = normalize(formula)
    if reordered:
        print("SFRIS-LIB: interpreted as %s (Hill notation)" % hill)
    if hill not in _load()['formulas']:
        msg = "SFRIS-LIB: %s not found in library." % hill
        alts = suggest(hill)
        if alts:
            msg += "\n           Did you mean: %s ?" % ', '.join(alts)
        raise LibraryError(msg)
    return hill

def _cmd_info():
    meta = info()
    print("SFRIS starting structure library")
    print("  library version : %s" % meta['library_version'])
    print("  generated by    : SFRIS %s / xTB %s (%s)"
          % (meta['sfris_version'], meta['xtb_version'], meta['method']))
    print("  parameters      : CutControl = %s" % meta['cut_control'])
    print("  derived from    : %s" % meta['source'])
    print("  coverage        : %s, up to 9 heavy atoms"
          % '/'.join(meta['elements']))
    print("  formulas        : %d" % meta['n_formulas'])
    print("  structures      : %d (top %d per formula)"
          % (meta['n_structures'], meta['top_n']))
    print()
    print("  %s" % meta['note'])
    print()
    print("  Cite Ramakrishnan (2014), Ruddigkeit (2012), Senthil (2021)")

def _cmd_list(formula):
    hill = _resolve(formula)
    rec = entry(hill)
    print()
    print("%s  (%d isomers found, top %d stored)"
          % (hill, rec['n_isomers'], rec['n_stored']))
    print()
    print("  rank  id     dE (kcal/mol)   E (Eh)")
    print("  " + "-" * 44)
    for s in rec['structures']:
        print("  %4d  %-5s %12.2f   %16.8f"
              % (s['rank'], s['id'], s['de'], s['e_xtb']))
    print()
    print(warning_text())

def _cmd_get(argv):
    if not argv:
        print(USAGE)
        return
    formula = argv[0]
    top = 1
    rest = argv[1:]
    i = 0
    while i < len(rest):
        if rest[i] in ('-n', '--top') and i + 1 < len(rest):
            try:
                top = int(rest[i + 1])
            except ValueError:
                raise LibraryError("invalid count: %s" % rest[i + 1])
            i += 2
        else:
            raise LibraryError("unknown option: %s" % rest[i])

    hill = _resolve(formula)
    structs = get(hill, top=top)

    print()
    for s in structs:
        name = "%s_lib_%02d.xyz" % (hill, s['rank'])
        with open(name, 'w') as fh:
            fh.write(to_xyz(hill, s))
        print("  wrote %s   dE = %.2f kcal/mol" % (name, s['de']))
    write_bibtex("SFRIS.bib")
    print("  wrote SFRIS.bib")
    print()
    print(warning_text())

def handle_lib(argv):
    """Entry point for `sfris lib ...`; argv is sys.argv[2:]."""
    if not argv:
        print(USAGE)
        return
    cmd = argv[0]
    try:
        if cmd == "info":
            _cmd_info()
        elif cmd == "list" and len(argv) >= 2:
            _cmd_list(argv[1])
        elif cmd == "get":
            _cmd_get(argv[1:])
        else:
            print(USAGE)
    except LibraryError as exc:
        print(exc)