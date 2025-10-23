#!/usr/bin/env python3
import os
import sys
from typing import Optional

# -------------------------------------------------------------------------------------------------
# Make parent importable if run as a script
current_dir = os.path.dirname(os.path.abspath(__file__))
top_level_module = os.path.abspath(os.path.join(current_dir, '..'))
if top_level_module not in sys.path:
    sys.path.insert(0, top_level_module)

# Project imports (match your updated modules)
from hardware.board_config import generateBoardConfig
from hardware.stm32.firmware_update import compileSTM32Flash
from robot.settings import settings_file_path, robot_path
from robot.control.bilbo_control_config import generate_default_config
from robot.hardware import (
    generateHardwareDefinition,
    writeHardwareDefinition,
)
from core.utils.files import fileExists, createFile
from core.utils.json_utils import readJSON, writeJSON
from hardware.shields.bilbo_shield_rev2 import generate_shield_config

# -------------------------------------------------------------------------------------------------


# =================================================================================================
# Helpers
# =================================================================================================
ALLOWED_BOARD_REVS = ('3', '4', '4.1')
ALLOWED_CM_TYPES = ('cm4', 'cm5')
ALLOWED_HW = ('bilbo1', 'bilbo2', 'mini', 'big')


def _normalize_board_rev(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    # allow inputs like 'rev4', '4', '4.1'
    if v.startswith('rev'):
        v = v[3:]
    if v in ALLOWED_BOARD_REVS:
        return v
    raise ValueError("Invalid board revision. Allowed: '3', '4', '4.1' (or 'rev3', 'rev4', 'rev4.1').")


def _normalize_cm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ALLOWED_CM_TYPES:
        return v
    raise ValueError("Invalid compute module type. Allowed: 'cm4' or 'cm5'.")


def _normalize_hw(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v == 'none':
        return None
    if v in ALLOWED_HW:
        return v
    raise ValueError("Invalid hardware config. Allowed: 'bilbo1', 'bilbo2', 'mini', 'big', or 'None' to skip.")


def _write_settings_file(robot_id: str,
                         board_rev: Optional[str],
                         cm_type: Optional[str],
                         hardware_config: Optional[str]) -> None:
    data = readJSON(settings_file_path) if fileExists(settings_file_path) else {}
    data['ID'] = robot_id
    data['board_rev'] = board_rev
    data['cm_type'] = cm_type
    data['hardware_config'] = hardware_config
    writeJSON(settings_file_path, data)

    # Make a robot marker file if missing
    robot_marker = f"{robot_path}/{robot_id}"
    if not fileExists(robot_marker):
        createFile(robot_marker)


def _generate_board(board_rev: Optional[str], cm_type: Optional[str]) -> None:
    if board_rev is None or cm_type is None:
        print("Skipping board config generation (board_rev or cm_type is None).")
        return
    # hardware.board_config.generateBoardConfig expects 'revX' style + cm type
    generateBoardConfig(f"rev{board_rev}", cm_type)
    print(f"Board config generated for revision {board_rev} and compute module {cm_type}.")


def _generate_hardware(hw: Optional[str]) -> None:
    if hw is None:
        print("Skipping hardware definition generation (hardware_config is None).")
        return
    # robot.hardware.generateHardwareDefinition expects uppercase choices
    hw_key = {
        'bilbo1': 'BILBO1',
        'bilbo2': 'BILBO2',
        'mini': 'MINI',
        'big': 'BIG',
    }[hw]
    hw_def = generateHardwareDefinition(hw_key)
    writeHardwareDefinition(hw_def)
    print(f"Hardware definition generated for {hw}.")


def _generate_control_config(hw: Optional[str]) -> None:
    if hw is None:
        print("Skipping control config generation (hardware_config is None).")
        return
    # Assuming generate_default_config now accepts the hardware id (bilbo1/bilbo2/mini/big)
    generate_default_config(hw)
    print(f"Default control config generated for hardware '{hw}'.")


def _maybe_read_settings():
    if fileExists(settings_file_path):
        try:
            return readJSON(settings_file_path)
        except Exception:
            return {}
    return {}


def _prompt(msg: str, default: Optional[str] = None) -> Optional[str]:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{msg}{suffix}: ").strip()
    if raw == "" and default is not None:
        return default
    if raw == "" and default is None:
        return None
    return raw


# =================================================================================================
# Unified setup flow
# =================================================================================================
def setup(board_rev: Optional[str] = None,
          cm_type: Optional[str] = None,
          hardware_config: Optional[str] = None,
          robot_id: Optional[str] = None,
          interactive: bool = False) -> None:
    """
    Unified setup. Any of the parameters can be None:
      - If interactive=True, missing values will be prompted.
      - If interactive=False, missing values may be pulled from settings file (if present).
      - If hardware_config is None, hardware + control config generation is skipped (by design).
      - If board_rev or cm_type is None, board config generation is skipped (by design).
    """
    # Load existing settings (if any)
    settings = _maybe_read_settings()

    # Merge precedence: explicit args > existing settings > None
    robot_id = robot_id or settings.get('ID')
    board_rev = board_rev or settings.get('board_rev')
    cm_type = cm_type or settings.get('cm_type')
    hardware_config = hardware_config or settings.get('hardware_config')

    # Interactive prompts (only for things still missing)
    if interactive:
        if settings:
            print(f"Found existing settings: ID={settings.get('ID')}, "
                  f"board_rev={settings.get('board_rev')}, cm_type={settings.get('cm_type')}, "
                  f"hardware_config={settings.get('hardware_config')}")
            use = _prompt("Use these settings? (y/n)", "y")
            if isinstance(use, str) and use.lower() != "y":
                # fall through to prompt everything
                robot_id = None
                board_rev = None
                cm_type = None
                hardware_config = None

        if robot_id is None:
            robot_id = _prompt("Enter robot ID (required)")
            if not robot_id:
                raise ValueError("Robot ID is required.")

        if board_rev is None:
            br = _prompt("Enter board revision ['3', '4', '4.1'] or leave blank to skip", None)
            board_rev = _normalize_board_rev(br) if br is not None else None

        if cm_type is None:
            cm = _prompt("Enter compute module ['cm4', 'cm5'] or leave blank to skip", None)
            cm_type = _normalize_cm(cm) if cm is not None else None

        if hardware_config is None:
            hw = _prompt("Enter hardware config ['bilbo1', 'bilbo2', 'mini', 'big'] or 'None' to skip", None)
            hardware_config = _normalize_hw(hw) if hw is not None else None
    else:
        # Non-interactive validations (only validate if provided)
        if robot_id is None:
            raise ValueError("Robot ID is required (or run with interactive=True).")
        board_rev = _normalize_board_rev(board_rev) if board_rev is not None else None
        cm_type = _normalize_cm(cm_type) if cm_type is not None else None
        hardware_config = _normalize_hw(hardware_config) if hardware_config is not None else None

    # Persist settings (even if some parts are None, by your spec)
    _write_settings_file(robot_id, board_rev, cm_type, hardware_config)

    # Generate things (each is independently skippable)
    _generate_board(board_rev, cm_type)
    _generate_hardware(hardware_config)
    _generate_control_config(hardware_config)

    # Optional firmware + shield steps (run regardless of skips above)
    compileSTM32Flash()
    generate_shield_config()

    print(f"Setup Complete: ID={robot_id}, board_rev={board_rev}, cm_type={cm_type}, hardware={hardware_config}")


# =================================================================================================
# CLI
# =================================================================================================
if __name__ == '__main__':
    """
    CLI usage:

      1) Interactive (prompts for anything missing):
            python setup.py
         or  python setup.py --interactive

      2) Non-interactive positional (None means skip that part):
            python setup.py <board_rev|None> <cm_type|None> <hardware|None> <robot_id>

         Examples:
            python setup.py 4 cm4 bilbo1 R2D2
            python setup.py None None mini R2D2
            python setup.py 4.1 cm5 None R2D2

      3) Non-interactive flags (order doesn’t matter):
            python setup.py --board 4 --cm cm4 --hw bilbo1 --id R2D2
            python setup.py --id R2D2 --hw None

      Notes:
        - Valid board_rev: 3, 4, 4.1 (you may also type rev4, rev4.1, etc.)
        - Valid cm_type: cm4, cm5
        - Valid hardware: bilbo1, bilbo2, mini, big, or None to skip hardware/control config generation
    """
    args = sys.argv[1:]


    # quick flag parsing to keep dependencies zero
    def _get_flag(flag: str) -> Optional[str]:
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return args[i + 1]
        return None


    use_interactive = ('--interactive' in args) or (len(args) == 0)

    if use_interactive:
        setup(interactive=True)

    else:
        # flags first
        br = _get_flag('--board')
        cm = _get_flag('--cm')
        hw = _get_flag('--hw')
        rid = _get_flag('--id')

        # if no flags, try positional form
        if br is None and cm is None and hw is None and rid is None:
            if len(args) not in (0, 4):
                print("Invalid arguments. Run with --interactive or provide 4 positional args: "
                      "<board_rev|None> <cm_type|None> <hardware|None> <robot_id>")
                sys.exit(1)
            br, cm, hw, rid = args  # type: ignore

        # Convert explicit 'None' strings to real None
        br = None if (br is not None and br.strip().lower() == 'none') else br
        cm = None if (cm is not None and cm.strip().lower() == 'none') else cm
        hw = None if (hw is not None and hw.strip().lower() == 'none') else hw


        setup(board_rev=br, cm_type=cm, hardware_config=hw, robot_id=rid, interactive=False)
