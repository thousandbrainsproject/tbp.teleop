# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hydra plugin that makes the `tbp.teleop` configs discoverable."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class TeleopSearchPathPlugin(SearchPathPlugin):
    """Adds the `tbp.teleop.conf` package to Hydra's config search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Append the `tbp.teleop` config package to the search path.

        Args:
            search_path: The Hydra config search path to modify.
        """
        search_path.append(provider="tbp.teleop", path="pkg://tbp.teleop.conf")
