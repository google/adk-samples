import ast
import datetime
import json
import re


def validate_code_safety(code: str) -> list:
    """
    Analyzes the code using AST to detect potentially unsafe operations.
    Returns a list of error messages if unsafe patterns are found.
    """
    errors = []

    # Exclude listed modules that should not be imported
    unsafe_modules = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pickle",
        "importlib",
        "socket",
        "http",
        "urllib",
        "requests",
    }

    # Exclude listed built-in functions
    unsafe_functions = {"eval", "exec", "open", "compile", "__import__"}

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax Error in generated code: {e}"]

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in unsafe_modules:
                    errors.append(
                        f"Security Violation: Import of restricted module '{alias.name}' is not allowed."
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in unsafe_modules:
                errors.append(
                    f"Security Violation: Import from restricted module '{node.module}' is not allowed."
                )

        # Check function calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in unsafe_functions:
                    errors.append(
                        f"Security Violation: Usage of restricted function '{node.func.id}' is not allowed."
                    )

    return errors


def run_simulation(policy_code: str, metadata: list) -> list:
    violations = []
    if not policy_code or policy_code.startswith("# API key not configured"):
        violations.append(
            {"policy": "Configuration Error", "violation": policy_code}
        )
        return violations

    if policy_code.startswith("# Error:"):
        violations.append(
            {"policy": "Execution Error", "violation": policy_code}
        )
        return violations

    # 1. Static Security Analysis
    security_errors = validate_code_safety(policy_code)
    if security_errors:
        for err in security_errors:
            violations.append(
                {"policy": "Security Violation", "violation": err}
            )
        return violations

    # 2. Prepare Restricted Environment
    # Only allow specific modules and built-ins
    safe_globals = {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "range": range,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "isinstance": isinstance,
        },
        "json": json,
        "re": re,
        "datetime": datetime,
    }

    try:
        # The generated code should define a function named check_policy
        # that takes metadata as an argument.
        # We execute the code in the restricted namespace.
        exec(policy_code, safe_globals)

        if "check_policy" in safe_globals:
            # The generated function returns a list of violations
            check_policy_func = safe_globals["check_policy"]
            violations = check_policy_func(metadata)
        else:
            violations.append(
                {
                    "policy": "Execution Error",
                    "violation": "Error executing policy code: 'check_policy' function not found.",
                }
            )
    except Exception as e:
        violations.append(
            {
                "policy": "Execution Error",
                "violation": f"Error executing policy code: {e}",
            }
        )

    return violations
