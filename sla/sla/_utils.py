"""Module for all utility functions.

"""

from typing import Dict, Optional, Type


def _get_all_subclasses_from_superclass(
    superclass: Type
) -> Dict[str, Optional[str]]:
    result = dict()
    for sb in superclass.__subclasses__():
        if sb.__name__[0] != "_":
            result.update({sb.__name__: sb.__doc__})
        else:
            result.update(_get_all_subclasses_from_superclass(sb))
    return result