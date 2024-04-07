import asyncio
import importlib
import inspect
import os
from contextlib import asynccontextmanager
from http import HTTPStatus

import fastapi
import uvicorn
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              CompletionRequest, ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)

#logit_list = [1, 7, 8, 11, 12, 13, 14, 30, 93, 98, 99, 102, 106, 107, 108, 110, 111, 119, 220, 222, 223, 224, 225, 226, 228, 230, 231, 236, 237, 238, 241, 242, 243, 247, 248, 250, 252, 253, 254, 256, 257, 319, 715, 854, 1719, 1773, 2073, 3224] 
logit_list = [3837, 4891, 5122, 5373, 5817, 6313, 6567, 6684, 7552, 8863, 8908, 8997, 9370, 9909, 10236, 10904, 10958, 11319, 11622, 12857, 13343, 14053, 14224, 14777, 15946, 16038, 16476, 16530, 16872, 17447, 17714, 17881, 17992, 18137, 18493, 18830, 19108, 20221, 20412, 20450, 20929, 21216, 21287, 21887, 21894, 22243, 22382, 22418, 22597, 22697, 22704, 23031, 23990, 24562, 25074, 26232, 26288, 26381, 26939, 27091, 27369, 27442, 27773, 28291, 28330, 29490, 29524, 29767, 30440, 30534, 30709, 31139, 31338, 31548, 31838, 31843, 31914, 32108, 32648, 32664, 32757, 32945, 33108, 33126, 33424, 33447, 34187, 34204, 34718, 34794, 35551, 35727, 35946, 36407, 36587, 36589, 36987, 36993, 37029, 37265, 38109, 38182, 38433, 39907, 40666, 40727, 40916, 41175, 41321, 41479, 42140, 42411, 43209, 43288, 43589, 43959, 44063, 44091, 44636, 44729, 44934, 45181, 45861, 45995, 46306, 46750, 46944, 47534, 47815, 47874, 48272, 48692, 48921, 48934, 49187, 49567, 49828, 50404, 50647, 50930, 51154, 52129, 52510, 52801, 52853, 53153, 53497, 54021, 54542, 54851, 55286, 56006, 56007, 56568, 56652, 57191, 57218, 57443, 57566, 57750, 58143, 58230, 58405, 58695, 58792, 59074, 59258, 59532, 59975, 60237, 60548, 60596, 60610, 60726, 60757, 62112, 62244, 62926, 62945, 63109, 63836, 64064, 64272, 64355, 64471, 64643, 65101, 66017, 66521, 67338, 68536, 69041, 69249, 70589, 71134, 71138, 71268, 71416, 71817, 72064, 72225, 72448, 73218, 73562, 73670, 75061, 75355, 75402, 75758, 77288, 77407, 77557, 78556, 80443, 81263, 81264, 81812, 82075, 82700, 82847, 84897, 85106, 85336, 85658, 86119, 87026, 87256, 87267, 87752, 88086, 88683, 88802, 89012, 90395, 90476, 90885, 90919, 91278, 91417, 91572, 92032, 92133, 93488, 93823, 94237, 94305, 94375, 95053, 96050, 96422, 97084, 97480, 97639, 97706, 97907, 98641, 99079, 99107, 99164, 99165, 99172, 99178, 99193, 99212, 99219, 99235, 99245, 99246, 99259, 99278, 99283, 99295, 99296, 99307, 99313, 99316, 99318, 99329, 99335, 99360, 99361, 99363, 99364, 99366, 99371, 99378, 99379, 99385, 99392, 99396, 99405, 99407, 99419, 99425, 99428, 99431, 99461, 99462, 99464, 99475, 99476, 99487, 99491, 99494, 99495, 99502, 99505, 99507, 99517, 99519, 99527, 99535, 99537, 99544, 99545, 99553, 99555, 99561, 99570, 99572, 99573, 99577, 99600, 99603, 99604, 99605, 99614, 99639, 99652, 99654, 99660, 99663, 99665, 99670, 99682, 99691, 99692, 99696, 99705, 99729, 99730, 99734, 99744, 99750, 99758, 99765, 99790, 99792, 99794, 99795, 99797, 99800, 99808, 99817, 99859, 99877, 99878, 99879, 99880, 99882, 99883, 99885, 99888, 99892, 99894, 99913, 99918, 99922, 99927, 99928, 99933, 99936, 99948, 99949, 99957, 99965, 99966, 99992, 99999, 100000, 100001, 100005, 100006, 100007, 100008, 100009, 100010, 100015, 100020, 100021, 100034, 100036, 100042, 100047, 100058, 100081, 100089, 100111, 100131, 100136, 100141, 100146, 100157, 100158, 100175, 100179, 100182, 100187, 100198, 100203, 100204, 100212, 100213, 100231, 100268, 100269, 100271, 100281, 100284, 100325, 100341, 100343, 100344, 100346, 100347, 100350, 100354, 100363, 100364, 100369, 100371, 100374, 100382, 100398, 100402, 100404, 100410, 100411, 100412, 100420, 100424, 100430, 100468, 100475, 100489, 100523, 100535, 100553, 100562, 100565, 100566, 100591, 100624, 100626, 100629, 100630, 100631, 100638, 100646, 100649, 100655, 100656, 100662, 100669, 100671, 100673, 100676, 100681, 100683, 100684, 100696, 100703, 100720, 100725, 100733, 100741, 100749, 100751, 100771, 100773, 100780, 100784, 100819, 100832, 100841, 100908, 100914, 100966, 101037, 101041, 101042, 101047, 101056, 101059, 101063, 101068, 101069, 101075, 101076, 101077, 101078, 101081, 101083, 101085, 101103, 101105, 101118, 101120, 101137, 101139, 101140, 101148, 101158, 101160, 101171, 101181, 101192, 101194, 101212, 101214, 101219, 101221, 101222, 101224, 101235, 101245, 101255, 101260, 101261, 101304, 101311, 101336, 101348, 101358, 101362, 101419, 101425, 101432, 101446, 101449, 101494, 101514, 101536, 101558, 101561, 101602, 101697, 101882, 101883, 101884, 101885, 101886, 101887, 101888, 101889, 101891, 101894, 101895, 101898, 101899, 101900, 101902, 101906, 101907, 101914, 101922, 101929, 101938, 101940, 101941, 101946, 101948, 101953, 101958, 101970, 101994, 101997, 102010, 102011, 102021, 102041, 102056, 102062, 102066, 102073, 102084, 102086, 102093, 102095, 102100, 102119, 102124, 102133, 102150, 102167, 102201, 102205, 102218, 102235, 102245, 102268, 102314, 102333, 102353, 102364, 102390, 102395, 102432, 102435, 102452, 102453, 102455, 102466, 102470, 102475, 102531, 102554, 102561, 102570, 102572, 102577, 102580, 102595, 102596, 102598, 102649, 102650, 102704, 102718, 102721, 102788, 102797, 102804, 102827, 102835, 102959, 102977, 103074, 103120, 103143, 103235, 103314, 103354, 103447, 103562, 103803, 103920, 103922, 103929, 103930, 103932, 103939, 103943, 103944, 103945, 103946, 103954, 103958, 103968, 103973, 103978, 103987, 103988, 103990, 103992, 103998, 103999, 104006, 104007, 104009, 104017, 104044, 104047, 104060, 104064, 104074, 104080, 104100, 104111, 104133, 104135, 104139, 104144, 104145, 104158, 104170, 104186, 104193, 104215, 104236, 104239, 104264, 104279, 104288, 104292, 104305, 104309, 104311, 104314, 104316, 104323, 104330, 104332, 104339, 104355, 104365, 104395, 104404, 104420, 104430, 104440, 104451, 104463, 104468, 104473, 104482, 104488, 104495, 104506, 104529, 104551, 104560, 104579, 104599, 104613, 104625, 104638, 104651, 104652, 104661, 104670, 104697, 104698, 104701, 104705, 104710, 104715, 104729, 104749, 104783, 104794, 104795, 104810, 104818, 104835, 104838, 104882, 104887, 104926, 104943, 104949, 104959, 104964, 104968, 104969, 105025, 105041, 105045, 105048, 105062, 105067, 105101, 105105, 105109, 105110, 105157, 105167, 105226, 105349, 105360, 105373, 105374, 105377, 105396, 105397, 105399, 105401, 105419, 105427, 105469, 105470, 105471, 105504, 105519, 105522, 105536, 105545, 105606, 105608, 105612, 105652, 105655, 105665, 105684, 105688, 105691, 105700, 105733, 105734, 105743, 105767, 105783, 105789, 105818, 105891, 105920, 105925, 105957, 105984, 105988, 106032, 106033, 106099, 106153, 106155, 106179, 106184, 106185, 106211, 106212, 106296, 106374, 106400, 106427, 106434, 106454, 106469, 106505, 106525, 106550, 106568, 106584, 106678, 106710, 106712, 106730, 106776, 106784, 106800, 106802, 106817, 106829, 106870, 106908, 106922, 106930, 106967, 107025, 107084, 107099, 107114, 107124, 107154, 107165, 107176, 107182, 107189, 107209, 107318, 107342, 107409, 107429, 107452, 107558, 107573, 107585, 107590, 107607, 107637, 107698, 107727, 107738, 107740, 107744, 107781, 107783, 107816, 107850, 107874, 107898, 107922, 107938, 107945, 107952, 107957, 107959, 107980, 107989, 108026, 108080, 108097, 108135, 108136, 108137, 108165, 108172, 108209, 108255, 108338, 108354, 108380, 108386, 108443, 108464, 108547, 108551, 108562, 108569, 108662, 108663, 108664, 108720, 108724, 108808, 108819, 108851, 108935, 108948, 109085, 109186, 109266, 109403, 109454, 109545, 109547, 109621, 109635, 109691, 109692, 109703, 109776, 109784, 109870, 109965, 110102, 110121, 110163, 110201, 110205, 110237, 110335, 110548, 110596, 110648, 110726, 110801, 110858, 110875, 111097, 111142, 111199, 111255, 111279, 111319, 111472, 111499, 111611, 111622, 111640, 111692, 111728, 111814, 111974, 111998, 112046, 112129, 112187, 112301, 112354, 112452, 112477, 112488, 112724, 112795, 112934, 113058, 113115, 113310, 113369, 113370, 113373, 113383, 113426, 113490, 113522, 113552, 113577, 113715, 113842, 113940, 114106, 114127, 114186, 114278, 114349, 114386, 114449, 114477, 114539, 114601, 114619, 114959, 115017, 115068, 115132, 115173, 115291, 115337, 115540, 115742, 115801, 115839, 115871, 116039, 116154, 116211, 116437, 116880, 116883, 117094, 117104, 117154, 117247, 117400, 117455, 117633, 117645, 117647, 117656, 117913, 118208, 118243, 118269, 118271, 118272, 118284, 118619, 119108, 119143, 119298, 119324, 119402]


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


app = fastapi.FastAPI(lifespan=lifespan)


def parse_args():
    parser = make_arg_parser()
    return parser.parse_args()


# Add prometheus asgi middleware to route /metrics requests
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    #token_range = logit_list
    request.logit_bias = {str(token) : -50 for token in logit_list}
    print(logit_list)
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    generator = await openai_serving_completion.create_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            root_path = "" if args.root_path is None else args.root_path
            if not request.url.path.startswith(f"{root_path}/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(f"Invalid middleware {middleware}. "
                             f"Must be a function or a class.")

    logger.info(f"vLLM API server version {vllm.__version__}")
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
    openai_serving_chat = OpenAIServingChat(engine, served_model,
                                            args.response_role,
                                            args.lora_modules,
                                            args.chat_template)
    openai_serving_completion = OpenAIServingCompletion(
        engine, served_model, args.lora_modules)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)
