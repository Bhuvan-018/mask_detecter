def commit_callback(commit):
    if commit.author_email == b"bhuvan87545@gmail.com":
        commit.author_name = b"Bhuvan-018"
        commit.author_email = b"191472861+Bhuvan-018@users.noreply.github.com"
    if commit.committer_email == b"bhuvan87545@gmail.com":
        commit.committer_name = b"Bhuvan-018"
        commit.committer_email = b"191472861+Bhuvan-018@users.noreply.github.com"