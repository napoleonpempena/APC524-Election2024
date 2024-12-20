import nox


@nox.session
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "serve" to serve.
    """

    session.install(".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")

@nox.session
def typecheck(session: nox.Session) -> None:
    """
    Run type checking using mypy.
    """
    session.install("mypy")
    session.install(
        "types-requests",
        "types-beautifulsoup4",
        "pandas-stubs"
    )
    session.run("mypy", "--ignore-missing-imports", "src")