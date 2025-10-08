# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import inspect

from src.utils.logging import get_logger

logger = get_logger("Eval runner scaffold")


def main(eval_name, args_eval, resume_preempt=False):
    logger.info(f"Running evaluation: {eval_name}")
    if eval_name.startswith("app."):
        import_path = f"{eval_name}.eval"
    else:
        import_path = f"evals.{eval_name}.eval"

    # Import module and log where the module and its main() are defined
    logger.info(f"Resolved import path: {import_path}")
    module = importlib.import_module(import_path)
    module_file = getattr(module, "__file__", None)
    if module_file:
        logger.info(f"Imported module file: {module_file}")
    else:
        logger.info("Imported module has no __file__ (namespace package or built-in)")

    main_func = getattr(module, "main", None)
    if main_func is None:
        logger.warning(f"Module {import_path} has no attribute 'main' to call")
    else:
        try:
            src_file = inspect.getsourcefile(main_func) or module_file
            src_lines = inspect.getsourcelines(main_func)[1]
            logger.info(f"Calling function: {import_path}.main defined in {src_file} at line {src_lines}")
        except Exception:
            logger.info(f"Calling function: {import_path}.main (source location unavailable)")

    return module.main(args_eval=args_eval, resume_preempt=resume_preempt)
