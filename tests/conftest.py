import os
import tempfile

# Keep test tickets out of data/tickets.db. Must be set before retailassist.tickets is imported.
os.environ["RETAILASSIST_TICKETS_DB"] = os.path.join(tempfile.mkdtemp(), "tickets.db")
