"""GetMeContext internal package.

Modules are imported explicitly by the app and tests (``utils.config``,
``utils.rag_chain``, …) rather than re-exported here, which keeps imports
light and avoids circular dependencies.
"""
