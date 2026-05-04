"""
Unit Test for MCP Server
========================
Tests:

"""

# Import implemented functions
try:
    from server import analyse_file_changes, get_pr_templates, mcp, suggest_template

    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)


class TestImplementation:
    """Test that the required functions are implemented"""

    def test_imports(self):
        """Test taht all required functions can be imported"""
        assert IMPORTS_SUCCESSFUL, "Failed to import required functions: Failed to"
        f" import required functions: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}"
        assert mcp
