from IPython import get_ipython


def is_notebook():
    """
    Check if the code is running in a Jupyter Notebook or not.

    Returns:
        bool: True if running in a Jupyter Notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False
