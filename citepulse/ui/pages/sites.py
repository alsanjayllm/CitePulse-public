import streamlit as st

from citepulse.db import get_session
from citepulse.settings import get_settings
from citepulse.sites import list_sites, remove_site
from citepulse.ui.components import render_header

render_header()
st.subheader("Sites")

with get_session() as session:
    sites = list_sites(session)
    max_sites = get_settings().max_sites

    st.caption(f"{len(sites)} / {max_sites} sites")

    if not sites:
        st.info("No sites tracked yet. Add one from the Run Audit page.")
    else:
        for site in sites:
            uid = str(site.id)
            pending_flag = f"pending-remove-{uid}"
            rm_id = f"rm-{uid}"
            ok_id = f"ok-{uid}"
            no_id = f"no-{uid}"

            col_url, col_added, col_remove = st.columns([3, 2, 1])
            col_url.write(site.url)
            col_added.write(site.created_at.strftime("%Y-%m-%d"))
            if col_remove.button("Remove", key=rm_id):
                st.session_state[pending_flag] = True

            if st.session_state.get(pending_flag):
                st.warning(
                    f"Remove {site.url} from your active sites? Its audit "
                    "history stays viewable from the History page."
                )
                col_confirm, col_cancel = st.columns(2)
                if col_confirm.button("Confirm remove", key=ok_id):
                    try:
                        remove_site(session, site.url)
                    except ValueError as exc:
                        st.error(str(exc))
                        st.session_state.pop(pending_flag, None)
                    else:
                        st.session_state.pop(pending_flag, None)
                        st.rerun()
                if col_cancel.button("Cancel", key=no_id):
                    st.session_state.pop(pending_flag, None)
                    st.rerun()
