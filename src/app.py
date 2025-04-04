# Chat app
import time
import json
import streamlit as st
from typing import Generator, Literal

# Custom classes
from src.crawlers import DeepCrawler
from src.retrieval import RAG
from src.models import LLM

# Configuration constants
from src.config import ChatConfig, CrawlerConfig, RAGConfig, LLMConfig

# Helper functions
from utils.helper_functions import connected_to_internet, extract_keywords, validate_url, get_timestamp, stream_response


def persist_state() -> None:
    """
    Keep session state consistent between reruns.
    """
    # Indicates whether app has fully loaded
    if "chat_loaded" not in st.session_state:
        st.session_state.chat_loaded = False

    # Initialize settings to hold submitted configuration
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "crawler": {
                # List all possible keys and set default values
                "enabled": False,
                "url": "",
                "relevant_search": False,
                "max_depth": 0,
                "max_pages": 1,
                "min_score": max(
                    0.0, 
                    min(0.8, round(CrawlerConfig.MIN_SCORE, 1))
                ),
                "kw_weight": max(
                    0.0, 
                    min(1.0, round(CrawlerConfig.KW_WEIGHT, 1))
                )
            },
            "rag": {
                # List all possible keys
                "search_function": "mmr",
                # These values are set dynamically depending on the crawling strategy
                "top_k": None,
                "fetch_k": None,
                "min_diversity": None,
                "min_similarity": None
            },
            "llm": {
                # List all possible keys
                "temperature": max(
                    0.0, 
                    min(1.0, round(LLMConfig.TEMPERATURE, 1))
                ),
                "max_tokens": -2
            }
        }

    # Store error from settings validation
    if "settings_error_message" not in st.session_state:
        st.session_state.settings_error_message = ""

    # Indicate whether settings submit is required
    if "settings_submit_required" not in st.session_state:
        st.session_state.settings_submit_required = False

    # Indicate whether initial settings were submitted
    if "initial_settings_submitted" not in st.session_state:
        st.session_state.initial_settings_submitted = False

    # Indicate whether app should rerun as settings are modified (initially set as True to run validations on entry)
    if "rerun_on_settings_change" not in st.session_state:
        st.session_state.rerun_on_settings_change = True

    # Indicate ongoing rerun not to validate the same settings configuration again
    if "settings_rerun_in_progress" not in st.session_state:
        st.session_state.settings_rerun_in_progress = False

    # Indicate whether comprehensive crawl (BFS) should be executed
    if "bfs_pending" not in st.session_state:
        st.session_state.bfs_pending = False

    # Indicate ongoing comprehensive crawl (BFS) execution
    if "bfs_in_progress" not in st.session_state:
        st.session_state.bfs_in_progress = False

    # Indicates that comprehensive crawl (BFS) is configured to run with identical settings as previous query-driven crawl (QDS)
    if "bfs_with_qds_settings" not in st.session_state:
        st.session_state.bfs_with_qds_settings = False

    # Indicate ongoing query-driven crawl (QDS) execution
    if "qds_in_progress" not in st.session_state:
        st.session_state.qds_in_progress = False

    # Indicate whether last crawl (either BFS or QDS) has finished execution
    if "last_crawl_completed" not in st.session_state:
        st.session_state.last_crawl_completed = True

    # Indicate whether response stream was written
    if "response_written" not in st.session_state:
        st.session_state.response_written = True

    # Initialize RAG
    if "rag" not in st.session_state:
        st.session_state.rag = RAG(
            collection_name=RAGConfig.COLLECTION_NAME,
            persist_directory=RAGConfig.PERSIST_DIRECTORY,
            embeddings_model=RAGConfig.EMBEDDINGS_MODEL,
            cleanup_collection=RAGConfig.CLEANUP_COLLECTION
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": ChatConfig.WELCOME_MESSAGE
        }]


def page_config(title: str) -> None:
    """
    Render page configuration.

    Parameters
    ----------
        title : str
            Title for the page.
    """
    # Basic page configuration
    st.set_page_config(page_title=title, page_icon="🤖")

    #  Title for the page
    st.title(title)


def message_history() -> None:
    """
    Loead chat messages from history on each rerun.
    """
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def ui_elements_enabled() -> bool:
    """
    Check if UI element should be rendered.

    Returns
    -------
        bool
            Whether UI element should be visible or disabled.
    """
    return st.session_state.chat_loaded and st.session_state.last_crawl_completed and st.session_state.response_written


def settings_change_callback() -> None:
    """
    Callback invoked when any of the settings change, forces app refresh to enable submit button.
    """
    # Rerun on settings change to refresh the sidebar
    st.session_state.rerun_on_settings_change = True


def submit_settings_callback() -> None:
    # Rerun on settings change to refresh the sidebar
    st.session_state.rerun_on_settings_change = True

    if st.session_state.bfs_with_qds_settings and st.session_state.crawler__repeat_bfs:
        # Repeat comprehensive crawl with the same settings
        st.session_state.bfs_pending = True

    # Rerun before BFS
    if connected_to_internet() and st.session_state.crawler__enabled and st.session_state.bfs_pending:
        # Indicate that crawl hasn't finished yet
        st.session_state.bfs_pending = False
        st.session_state.last_crawl_completed = False
        st.session_state.bfs_in_progress = True


def basic_settings() -> None:
    """
    Basic settings widget.
    """
    with st.expander(label="Basic configuration", icon=":material/settings:", expanded=True):
        options_map = {
            False: ":material/public_off: Existing knowledge",
            True: ":material/travel_explore: Search websites"
        }

        if connected_to_internet():
            st.pills(
                label="Retrieval mode",
                options=options_map.keys(),
                format_func=lambda x: options_map[x],
                key="crawler__enabled",
                selection_mode="single",
                disabled=not ui_elements_enabled(),
                default=False,
                on_change=settings_change_callback
            )
        
            if st.session_state.crawler__enabled:
                st.caption(
                    "Crawl websites to make the answers **more contextual**"
                )

            else:
                st.caption(
                    "Answer general questions with **existing knowledge**"
                )
        
        else:
            # This section can be displayed AFTER crawler's settings are validated and submitted
            # Explicitly disable crawler to prevent its functionality without internet connection
            st.session_state.settings["crawler"]["enabled"] = False
            
            st.session_state.crawler__enabled = st.pills(
                label="Retrieval mode",
                options=options_map.keys(),
                format_func=lambda x: options_map[x],
                selection_mode="single",
                default=False,
                disabled=True
            )

            st.caption(
                "Answer general questions with **existing knowledge**"
            )

            st.warning("No internet connection, search mode unavailable")

            retry_connection = st.button(
                label="Retry the connection",
                type="tertiary",
                icon=":material/refresh:",
                disabled=not ui_elements_enabled(),
            )

            
            if retry_connection:
                # Display progress bar
                connection_text = "Checking internet connectivity"
                connection_bar = st.progress(0, text=connection_text)

                # Imitate progress of the execution
                for percent_complete in range(100):
                    time.sleep(0.002)
                    connection_bar.progress(percent_complete + 1, text=connection_text)

                # Remove progress bar after completion of the loop
                time.sleep(1)
                connection_bar.empty()


def crawler_settings() -> None:
    """
    Crawler settings widget.
    """
    with st.expander(label="Crawler configuration", icon=":material/footprint:", expanded=st.session_state.crawler__enabled):
        st.text_input(
            label="Website URL",
            placeholder="https://YourWebsiteGoesHere.com",
            key="crawler__url",
            disabled=not (
                st.session_state.crawler__enabled and ui_elements_enabled()
            ),
            on_change=settings_change_callback
        )

        options_map = {
            False: "Comprehensive",
            True: "Query-driven"
        }

        st.segmented_control(
            label="Crawl strategy",
            options=options_map.keys(),
            format_func=lambda k: options_map[k],
            selection_mode="single",
            key="crawler__relevant_search",
            default=False,
            disabled=not (
                st.session_state.crawler__enabled and ui_elements_enabled()
            ),
            on_change=settings_change_callback
        )

        if st.session_state.crawler__relevant_search:
            st.caption(
                body="""
                            Vector store is populated for each prompt to **prioritize the most relevant pages**
                            - **Slower execution** as website is crawled for **each prompt**
                            - Smaller, query-oriented knowledgebase is **more resource efficient**
                        """
            )

        else:
            st.caption(
                body="""
                            Vector store is populated initially **without prioritizing pages**
                            - **Faster execution** as website is crawled **only once**
                            - Comprehensive knowledgebase is **more suitable for general FAQs**
                        """
            )

        st.select_slider(
            label="Levels to crawl deeper",
            options=[i for i in range(0, max(1, CrawlerConfig.MAX_DEPTH) + 1)],
            format_func=lambda x:
                "Starting page only"
                if not x else int(x),
            key="crawler__max_depth",
            value=st.session_state.settings["crawler"]["max_depth"],
            disabled=not (
                st.session_state.crawler__enabled and ui_elements_enabled()
            ),
            on_change=settings_change_callback
        )

        if st.session_state.crawler__enabled:
            st.warning(
                "Beware of values greater than 3, which can exponentially increase crawl size"
            )

        options = [
            i for i in range(1, max(1, CrawlerConfig.MAX_PAGES) + 1)
        ] + [None,]

        options_map = {
            1: "Starting page only"
        }

        st.select_slider(
            label="Total number of pages to crawl",
            options=options,
            format_func=lambda x: "All" if not x else options_map.get(x, x),
            value=st.session_state.settings["crawler"]["max_pages"],
            key="crawler__max_pages",
            disabled=not (
                st.session_state.crawler__enabled and ui_elements_enabled()
            ),
            on_change=settings_change_callback
        )

        if st.session_state.crawler__enabled:
            if st.session_state.crawler__relevant_search:
                st.info(
                    body="Keep maximum number of pages **low**"
                )

            else:
                st.info(
                    body="Keep maximum number of pages **high** or use **all**"
                )

        if st.session_state.crawler__relevant_search:
            # Pop conditional control values that are not used
            st.session_state.pop("crawler__min_score", None)
            st.session_state.pop("crawler__repeat_bfs", None)

            options = [i / 10 for i in range(11)]
            options_map = {
                0.0: "No prioritization",
                1.0: "Only pages matching query"
            }

            st.select_slider(
                label="Prioritize keywords in overall scores",
                options=options,
                format_func=lambda x: options_map.get(x, x),
                key="crawler__kw_weight",
                value=st.session_state.settings["crawler"]["kw_weight"],
                disabled=not (
                    st.session_state.crawler__enabled and ui_elements_enabled()
                ),
                on_change=settings_change_callback
            )

        else:
            # Pop conditional control values that are not used
            st.session_state.pop("crawler__kw_weight", None)

            options = [i / 10 for i in range(9)]
            options_map = {
                0.0: "Any page",
                0.8: "High-score pages only"
            }

            st.select_slider(
                label="Minimum score for pages to be crawled",
                options=options,
                format_func=lambda x: options_map.get(x, x),
                key="crawler__min_score",
                value=st.session_state.settings["crawler"]["min_score"],
                disabled=not (
                    st.session_state.crawler__enabled and ui_elements_enabled()
                ),
                on_change=settings_change_callback
            )

            if st.session_state.bfs_with_qds_settings:
                if st.session_state.crawler__enabled:
                    st.warning(
                        "Settings are identical to the previous query-driven search"
                    )

                st.caption(
                    "Would you like to run comprehensive search **without adjustments**?"
                )

                st.checkbox(
                    label="Confirm subsequent iteration",
                    value=False,
                    key="crawler__repeat_bfs",
                    disabled=not (
                        st.session_state.crawler__enabled and ui_elements_enabled()
                    ),
                    on_change=settings_change_callback
                )


def rag_settings() -> None:
    """
    RAG settings widget.
    """
    with st.expander(label="RAG configuration", icon=":material/library_books:"):
        options_map = {
            "mmr": "MMR",
            "similarity_score_threshold": "Similarity"
        }

        st.segmented_control(
            label="Search function",
            options=options_map.keys(),
            key="rag__search_function",
            format_func=lambda x: options_map[x],
            disabled=not ui_elements_enabled(),
            default=st.session_state.settings["rag"]["search_function"],
            on_change=settings_change_callback
        )

        st.caption(
            """
                **Similarity** search retrieves **all the closest matches**,
                aditionally **MMR** considers their novelty to **avoid redundancy**.
                """
        )

        # Use as default value
        settings_value = st.session_state.settings["rag"]["top_k"]

        st.select_slider(
            label="Documents returned to the LLM",
            options=[i for i in range(1, max(3, RAGConfig.TOP_K) + 1)],
            key="rag__top_k",
            disabled=not ui_elements_enabled(),
            value=settings_value if settings_value is not None else 2,
            on_change=settings_change_callback
        )

        if st.session_state.rag__search_function == "mmr":

            # Pop conditional control values that are not used
            st.session_state.pop("rag__min_similarity", None)

            # Use as default value
            settings_value = st.session_state.settings["rag"]["fetch_k"]

            st.select_slider(
                label="Documents retrieved before filtering",
                options=[i for i in range(10, max(30, RAGConfig.FETCH_K) + 1)],
                key="rag__fetch_k",
                disabled=not (
                    st.session_state.chat_loaded and
                    st.session_state.last_crawl_completed and
                    st.session_state.response_written
                ),
                value=settings_value if settings_value is not None else 20,
                on_change=settings_change_callback
            )

            options_map = {
                0.0: "Diversity",
                1.0: "Relevance"
            }

            # Use as default value
            settings_value = st.session_state.settings["rag"]["min_diversity"]

            st.select_slider(
                label="Trade-off between relevance & diversity",
                options=[i / 10 for i in range(11)],
                format_func=lambda x: options_map.get(x, x),
                key="rag__min_diversity",
                disabled=not (
                    st.session_state.chat_loaded and
                    st.session_state.last_crawl_completed and
                    st.session_state.response_written
                ),
                value=settings_value if settings_value is not None else 0.6,
                on_change=settings_change_callback
            )

        else:

            # Pop conditional control values that are not used
            st.session_state.pop("rag__fetch_k", None)
            st.session_state.pop("rag__min_diversity", None)

            options = [i / 10 for i in range(11)]

            options_map = {
                0.0: "Unrelated",
                1.0: "Identical"
            }

            # Use as default value
            settings_value = st.session_state.settings["rag"]["min_similarity"]

            st.select_slider(
                label="Minimum similarity required",
                options=options,
                format_func=lambda x: options_map.get(x, x),
                key="rag__min_similarity",
                disabled=not (
                    st.session_state.chat_loaded and
                    st.session_state.last_crawl_completed and
                    st.session_state.response_written
                ),
                value=settings_value if settings_value is not None else 0.3,
                on_change=settings_change_callback
            )

            st.warning(
                body="Consider including more documents by lowering this value"
            )


def llm_settings() -> None:
    """
    LLM settings widget.
    """
    with st.expander(label="LLM configuration", icon=":material/network_intelligence:"):
        options_map = {
            0.0: "Deterministic",
            1.0: "Creative"
        }

        st.select_slider(
            label="Adjust response creativity",
            options=[i / 10 for i in range(11)],
            format_func=lambda x: options_map.get(x, x),
            key="llm__temperature",
            disabled=not ui_elements_enabled(),
            value=st.session_state.settings["llm"]["temperature"],
            on_change=settings_change_callback
        )

        if st.session_state.llm__temperature > 0.5:
            st.info(
                "Increasing the temperature makes the answers **more creative**"
            )

        else:
            st.info(
                "Decreasing the temperature makes the answers **more deterministic**"
            )

        max_tokens = round(max(128, LLMConfig.MAX_TOKENS) / 128)

        st.select_slider(
            label="Tokens in the generated response",
            options=[-2, ] + [
                i for i in range(128, (max_tokens * 128) + 1, 128)
            ],
            format_func=lambda x:
                "Fill context" if x == -2
                else int(x),
            key="llm__max_tokens",
            disabled=not ui_elements_enabled(),
            value=st.session_state.settings["llm"]["max_tokens"],
            on_change=settings_change_callback
        )


def sidebar() -> None:
    """
    Render sidebar widget. 
    """
    # Keep in mind that sidebars do not always reload and explicit re-runs are necessary to prevent it from displaying stale information.
    with st.sidebar:

        # Render settings form
        with st.container(border=True):

            # Configure header and caption
            st.header("Settings")
            st.caption("Modify settings to control behaviour of your assistant")

            # Display status message
            if not st.session_state.chat_loaded:
                st.warning("Please wait until chat widget is fully loaded")
            elif st.session_state.settings_error_message:
                st.error(st.session_state.settings_error_message)
            else:
                st.success("Settings configured successfully")

            # Render basic settings
            basic_settings()

            # Render crawler settings
            crawler_settings()

            # Render RAG settings
            rag_settings()

            # Render LLM settings
            llm_settings()

            # Custom form submit button
            st.session_state.submit_settings_clicked = st.button(
                label="Submit",
                type="primary",
                key="submit_settings",
                disabled=not (
                    st.session_state.settings_submit_required and
                    st.session_state.last_crawl_completed and
                    st.session_state.response_written
                ),
                on_click=submit_settings_callback
            )

        # Render options
        with st.container(border=True):

            # Configure header and caption
            st.header("Options")
            st.caption(
                "Use buttons to download chat transcript or clear message history"
            )

            # Add columns to store buttons
            col1, col2 = st.columns(2)

            with col1:
                # Serialize messages
                serialized_messages = json.dumps(
                    st.session_state["messages"]
                )

                # Download chat transcript
                st.download_button(
                    label="Download transcript",
                    icon=":material/download:",
                    # Encode serialized data
                    data=serialized_messages.encode(),
                    file_name=f"chat_conversation_{get_timestamp()}.json",
                    mime="application/json",
                    use_container_width=True,
                    disabled=(
                        len(st.session_state.messages) <= 1 or not (
                            st.session_state.last_crawl_completed and
                            st.session_state.response_written
                        )
                    )
                )

            with col2:
                # Remove message history
                message_cleanup = st.button(
                    label="Clear messages",
                    icon=":material/delete:",
                    use_container_width=True,
                    disabled=(
                        len(st.session_state.messages) <= 1 or not (
                            st.session_state.last_crawl_completed and
                            st.session_state.response_written
                        )
                    )
                )

            if message_cleanup:
                # Display progress bar
                cleanup_text = "Cleanup in progress. Please wait."
                cleanup_bar = st.progress(
                    value=0,
                    text=cleanup_text
                )

                # Imitate progress of the execution
                for percent_complete in range(100):
                    time.sleep(0.01)
                    cleanup_bar.progress(
                        value=percent_complete + 1,
                        text=cleanup_text
                    )

                # Remove progress bar after completion of the loop
                time.sleep(1)
                cleanup_bar.empty()

                # Remove messages
                del st.session_state["messages"]

                # Force rerun to refresh messages
                st.rerun()


def debug_settings(primary_key: Literal["crawler", "rag", "llm"]) -> None:
    """
    Debug settings.

    Parameters
    ----------
        primary_key : Literal["crawler", "rag", "llm"]
            Key to the appropriate configuration.
    """
    print(
        f"\n=== {primary_key.upper()} modified settings: {[
            (foreign_key, st.session_state.get(f"{primary_key}__{foreign_key}", None), value) for foreign_key, value in st.session_state.settings[primary_key].items()
        ]} ==="
    )
    print(
        f"=== Initial settings submitted: {st.session_state.initial_settings_submitted} ==="
    )
    print(
        f"=== Submit button clicked: {st.session_state.submit_settings_clicked} ==="
    )
    print(
        f"=== Rerun on settings change: {st.session_state.rerun_on_settings_change} ==="
    )
    print(
        f"=== Submit required: {st.session_state.settings_submit_required} ==="
    )
    if primary_key == "crawler":
        print(
            f"=== BFS pending: {st.session_state.bfs_pending} ==="
        )
        print(
            f"=== BFS in progress: {st.session_state.bfs_in_progress} ==="
        )
        print(
            f"=== Last crawl completed: {st.session_state.last_crawl_completed} ==="
        )
        print(
            f"=== BFS with QDS settings: {st.session_state.bfs_with_qds_settings} ==="
        )
    print("\n")


def modified_settings(primary_key: Literal["crawler", "rag", "llm"]) -> Generator[str]:
    """
    Retrieve list of keys associated with the settings, which were modified.

    Parameters
    ----------
        primary_key : Literal["crawler", "rag", "llm"]
            Key to the appropriate configuration.

    Returns
    -------
        Generator[str]
            Yields keys corresponding to the modified settings.
    """
    # Check if submit is required
    for foreign_key, value in st.session_state.settings[primary_key].items():

        # Get value from the control
        control_value = st.session_state.get(
            f"{primary_key}__{foreign_key}",
            None
        )

        # Settings were modified
        if control_value is not None and control_value != value:

            # Yield modified key
            yield foreign_key


def llm_settings_submitted() -> bool:
    """
    Handle manual submission of the LLM's settings.

    Returns
    -------
        bool
            Indicates whether any modified settings are submitted.
    """
    # Get list of modified keys
    modified_keys = [key for key in modified_settings("llm")]

    # LLM does not exist or submit is required
    if "llm" not in st.session_state or modified_keys:

        # Flag required submit
        st.session_state.settings_submit_required = True

        # Changes are not submitted (skip initial settings)
        if st.session_state.initial_settings_submitted and not st.session_state.submit_settings_clicked:

            # Update error message
            st.session_state.settings_error_message = "Please submit settings"

            # Flag ongoing rerun to skip settings validation and show error status
            st.session_state.settings_rerun_in_progress = True

            # Indicate error
            return False

        # Update settings
        for key in modified_keys:

            # Assign most recent value from the control
            st.session_state.settings["llm"][key] = st.session_state[f"llm__{key}"]

        # Initialize LLM with modified settings
        st.session_state.llm = LLM(
            system_prompt=LLMConfig.SYSTEM_PROMPT,
            model_variant=LLMConfig.MODEL_VARIANT,
            temperature=st.session_state.settings["llm"]["temperature"],
            max_tokens=st.session_state.settings["llm"]["max_tokens"]
        )

    # Indicate success
    return True


def rag_settings_submitted() -> bool:
    """
    Handle manual submission of the RAG's settings.

    Returns
    -------
        bool
            Indicates whether any modified settings are submitted.
    """
    # Get list of modified keys
    modified_keys = [key for key in modified_settings("rag")]

    # LLM does not exist or submit is required
    if modified_keys:

        # Flag required submit
        st.session_state.settings_submit_required = True

        # Changes are not submitted (skip initial settings)
        if st.session_state.initial_settings_submitted and not st.session_state.submit_settings_clicked:

            # Update error message
            st.session_state.settings_error_message = "Please submit settings"

            # Flag ongoing rerun to skip settings validation and show error status
            st.session_state.settings_rerun_in_progress = True

            # Indicate error
            return False

        # Update settings
        for key in modified_keys:

            # Assign most recent value from the control
            st.session_state.settings["rag"][key] = st.session_state[f"rag__{key}"]

    # Indicate success
    return True


def crawler_settings_submitted() -> bool:
    """
    Handle manual submission of the crawler's settings.

    Returns
    -------
        bool
            Indicates whether any modified settings are submitted.
    """
    # Revert flags on entry
    st.session_state.bfs_pending = False
    st.session_state.bfs_with_qds_settings = False

    # Crawling mode is enabled
    if st.session_state.crawler__enabled:

        # Website URL must not be empty
        if not st.session_state.crawler__url:

            # Update error message
            st.session_state.settings_error_message = "Please configure website URL"

            # Flag ongoing rerun to skip settings validation and show error status
            st.session_state.settings_rerun_in_progress = True

            # Indicate error
            return False

        # Retrieve connection errors
        connection_error = validate_url(st.session_state.crawler__url)

        if connection_error:
            # Update error message
            st.session_state.settings_error_message = f"Connection with the URL failed: {connection_error}"

            # Flag ongoing rerun to skip settings validation and show error status
            st.session_state.settings_rerun_in_progress = True

            # Indicate error
            return False

        # Store modified keys in set for faster retrieval
        modified_keys = {key for key in modified_settings("crawler")}

        # Query-driven crawl runs for each prompt, but comprehensive crawl runs conditionally
        if not st.session_state.crawler__relevant_search:
            
            # Perform comprehensive crawl as long as any associated settings changed
            if any(key in modified_keys for key in {"url", "max_depth", "max_pages", "min_score"}):

                # Indicate that comprehensive crawl can be executed
                st.session_state.bfs_pending = True
            
            # Parameters are the same as in the previous query-driven search
            elif st.session_state.settings["crawler"]["relevant_search"]:
                
                # User confirms whether to repeat comprehensive crawl with the same settings
                st.session_state.bfs_with_qds_settings = True
        
        # Crawler's settings were modified
        if modified_keys:

            # Submit is required
            st.session_state.settings_submit_required = True

            # Changes are not submitted
            if not st.session_state.submit_settings_clicked:

                # Update error message
                st.session_state.settings_error_message = "Please submit settings"

                # Flag ongoing rerun to skip settings validation and show error status
                st.session_state.settings_rerun_in_progress = True

                # Indicate error
                return False

            if "crawler" not in st.session_state or st.session_state.settings["crawler"]["url"] != st.session_state.crawler__url:
                # Initialize crawler
                st.session_state.crawler = DeepCrawler(
                    start_url=st.session_state.crawler__url
                )

            # Update settings
            for key in modified_keys:

                # Assign most recent value from the control
                st.session_state.settings["crawler"][key] = st.session_state[f"crawler__{key}"]

        # Indicate success
        return True

    # Crawling mode is disabled
    st.session_state.settings["crawler"]["enabled"] = False

    # Indicate success
    return True


def settings_configured() -> bool:
    """
    Validate settings configuration.

    Returns
    -------
        bool
            Indicates whether chat configuration is ready.
    """
    # Refresh flags before checking for settings that require submit
    st.session_state.settings_error_message = ""
    st.session_state.settings_submit_required = False

    if not llm_settings_submitted():
        # Some modified LLM's settigns were not submitted
        return False

    if not rag_settings_submitted():
        # Some modified RAG's settigns were not submitted
        return False

    if not crawler_settings_submitted():
        # Some modified crawler's settigns were not submitted
        return False

    # Revert flags as soon as settings are submitted
    st.session_state.settings_error_message = ""
    st.session_state.settings_submit_required = False
    st.session_state.initial_settings_submitted = True

    # Indicate success
    return True


def chat_input_callback() -> None:
    """
    Callback function when chat input is clicked.
    """
    # Indicate that stream response is going to be written soon
    st.session_state.response_written = False

    if connected_to_internet() and st.session_state.settings["crawler"]["enabled"] and st.session_state.settings["crawler"]["relevant_search"]:
        # Indicate that query-driven crawl will be executed soon
        st.session_state.qds_in_progress = True
        st.session_state.last_crawl_completed = False


async def retrieve_knowledgebase(keywords: list[str] = []) -> None:
    """
    Crawl website and populate vector store with the results.

    Parameters
    ----------
        keywords : list[str]
            List of keywords to prioritize the most relevant pages.
    """
    # Populate collection with vectors
    async for document in st.session_state.crawler.crawl(
        # Streaming mode is recommended for real-time applications
        stream_mode=True,
        max_depth=st.session_state.settings["crawler"]["max_depth"],
        max_pages=st.session_state.settings["crawler"]["max_pages"],
        min_score=st.session_state.settings["crawler"]["min_score"],
        kw_weight=st.session_state.settings["crawler"]["kw_weight"],
        kw_list=keywords
    ):
        if document.page_content:
            await st.session_state.rag.add_to_collection(
                document,
                # Do not exceed window size of 2048
                window_size=max(128, min(2048, RAGConfig.WINDOW_SIZE)),
                # Do not exceed 30% overlap
                window_overlap=max(0.0, min(0.3, RAGConfig.WINDOW_OVERLAP))
            )


async def chat_response(user_prompt: str) -> None:
    """
    Respond to the user's prompt.

    Parameters
    ----------
        user_prompt : str
            Prompt entered using chat input widget.
    """
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Run query-driven crawl (QDS) on each prompt
    if st.session_state.qds_in_progress:
        with st.spinner("Populating collection"):
            # Extract keywords from the prompt
            keywords = extract_keywords(
                text=user_prompt,
                ngram_size=min(3, max(1, CrawlerConfig.KW_NGRAM)),
                dedup_factor=min(1.0, max(0.0, CrawlerConfig.KW_DEDUP)),
                kw_prop=min(0.3, max(0.0, CrawlerConfig.KW_PROP)),
                kw_num=max(1, CrawlerConfig.KW_NUM)
            )

            # Notice that keywords are passed for the relevancy scorer
            await retrieve_knowledgebase(keywords)

        # Update flags as soon as query-driven crawl finishes
        st.session_state.qds_in_progress = False
        st.session_state.last_crawl_completed = True

    with st.spinner("Generating response"):
        # Retrieve context with RAG
        context = st.session_state.rag.get_context(
            query=user_prompt,
            search_function=st.session_state.settings["rag"]["search_function"],
            top_k=st.session_state.settings["rag"]["top_k"],
            fetch_k=st.session_state.settings["rag"]["fetch_k"],
            min_diversity=st.session_state.settings["rag"]["min_diversity"],
            min_similarity=st.session_state.settings["rag"]["min_similarity"]
        )

        # Generate response with LLM
        llm_response = st.session_state.llm.respond(user_prompt, context)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        chat_response = st.write_stream(
            stream_response(
                text=llm_response,
                seconds=max(0.0, ChatConfig.STREAM_INTERVAL)
            )
        )

    # Update flags as soon as writing stream response finishes
    st.session_state.response_written = True

    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": chat_response
    })


async def chat_widget() -> None:
    """
    Render chat widget.
    """
    # Refrain from verifying settings' integrity before chat loading is complete
    if st.session_state.chat_loaded:

        # Skip settings validation after rerun to avoid numerous screen renders
        if st.session_state.rerun_on_settings_change and not st.session_state.settings_rerun_in_progress:

            # Chat is not yet configured
            if not settings_configured():
                
                # Rerun to display error status, disable chat input and options, and enable submit button if required
                st.rerun()

            # Prevent subsequent reruns
            st.session_state.rerun_on_settings_change = False

            # Flag ongoing rerun to disable submit button and show success status after settings were submitted
            st.session_state.settings_rerun_in_progress = True

            # Rerun to display success status, enable chat input and options, and disable submit button
            st.rerun()

        # Indicate that rerun has completed
        st.session_state.settings_rerun_in_progress = False
    
    # Load message history
    message_history()

    # Display chat input widget
    if user_prompt := st.chat_input(
        placeholder="Ask anything",
        on_submit=chat_input_callback,
        disabled=(
            st.session_state.settings_error_message != "" or not ui_elements_enabled()
        ),
    ):
        # Reppond to the prompt
        await chat_response(user_prompt)

        # Force rerun to refresh messages (prevent stale message state for the download option)
        st.rerun()

    # Perform comprehesive crawl (BFS) after submitting modified settings
    if st.session_state.bfs_in_progress:
        with st.spinner("Populating collection"):
            # Notice that no keywords are extracted
            await retrieve_knowledgebase([])

        # Update flags as soon as comprehensive crawl finishes
        st.session_state.bfs_in_progress = False
        st.session_state.last_crawl_completed = True
        st.session_state.bfs_with_qds_settings = False

        # Rerun to enable all previously disabled controls including chat input
        st.rerun()

    # Update flags as soon as chat widget is loaded
    if not st.session_state.chat_loaded:
        # Rerun to refresh the sidebar and load settings
        st.session_state.chat_loaded = True
        time.sleep(2)
        st.rerun()


async def run_chat() -> None:
    """
    Run chat application.
    """
    # Keep session state consistent between reruns
    persist_state()

    # Basic page configuration
    page_config(title="Talk to your Website")

    # Render sidebar
    sidebar()

    # Render chat widget
    await chat_widget()
