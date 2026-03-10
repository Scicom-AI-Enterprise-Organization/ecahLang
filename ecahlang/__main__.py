from ecahlang.main import app, args
import uvicorn

uvicorn.run(
    app,
    host=args.host,
    port=args.port,
    log_level=args.loglevel.lower(),
    access_log=True,
    loop="uvloop",
)
