mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\

[theme]\n\
base="light"\n\
\n\
" > ~/.streamlit/config.toml
