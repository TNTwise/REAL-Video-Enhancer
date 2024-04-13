#!/usr/bin/env python

import collections
import os
import re
import typing

import cupy
import torch

##########################################################


objCudacache = {}


def cuda_int32(intIn:int):
    return cupy.int32(intIn)
# end


def cuda_float32(fltIn:float):
    return cupy.float32(fltIn)
# end


def cuda_kernel(strFunction:str, strKernel:str, objVariables:typing.Dict):
    if 'device' not in objCudacache:
        objCudacache['device'] = torch.cuda.get_device_name()
    # end

    strKey = strFunction

    for strVariable in objVariables:
        objValue = objVariables[strVariable]

        strKey += strVariable

        if objValue is None:
            continue

        elif type(objValue) == int:
            strKey += str(objValue)

        elif type(objValue) == float:
            strKey += str(objValue)

        elif type(objValue) == bool:
            strKey += str(objValue)

        elif type(objValue) == str:
            strKey += objValue

        elif type(objValue) == torch.Tensor:
            strKey += str(objValue.dtype)
            strKey += str(objValue.shape)
            strKey += str(objValue.stride())

        elif True:
            print(strVariable, type(objValue))
            assert(False)

        # end
    # end

    strKey += objCudacache['device']

    if strKey not in objCudacache:
        for strVariable in objVariables:
            objValue = objVariables[strVariable]

            if objValue is None:
                continue

            elif type(objValue) == int:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == float:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == bool:
                strKernel = strKernel.replace('{{' + strVariable + '}}', str(objValue))

            elif type(objValue) == str:
                strKernel = strKernel.replace('{{' + strVariable + '}}', objValue)

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.uint8:
                strKernel = strKernel.replace('{{type}}', 'unsigned char')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float16:
                strKernel = strKernel.replace('{{type}}', 'half')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float32:
                strKernel = strKernel.replace('{{type}}', 'float')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.float64:
                strKernel = strKernel.replace('{{type}}', 'double')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int32:
                strKernel = strKernel.replace('{{type}}', 'int')

            elif type(objValue) == torch.Tensor and objValue.dtype == torch.int64:
                strKernel = strKernel.replace('{{type}}', 'long')

            elif type(objValue) == torch.Tensor:
                print(strVariable, objValue.dtype)
                assert(False)

            elif True:
                print(strVariable, type(objValue))
                assert(False)

            # end
        # end

        while True:
            objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

            if objMatch is None:
                break
            # end

            intArg = int(objMatch.group(2))

            strTensor = objMatch.group(4)
            intSizes = objVariables[strTensor].size()

            strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg] if torch.is_tensor(intSizes[intArg]) == False else intSizes[intArg].item()))
        # end

        while True:
            objMatch = re.search('(OFFSET_)([0-4])(\()', strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')

            assert(intArgs == len(strArgs) - 1)

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')
            # end

            strKernel = strKernel.replace('OFFSET_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', '(' + str.join('+', strIndex) + ')')
        # end

        while True:
            objMatch = re.search('(VALUE_)([0-4])(\()', strKernel)

            if objMatch is None:
                break
            # end

            intStart = objMatch.span()[1]
            intStop = objMatch.span()[1]
            intParentheses = 1

            while True:
                intParentheses += 1 if strKernel[intStop] == '(' else 0
                intParentheses -= 1 if strKernel[intStop] == ')' else 0

                if intParentheses == 0:
                    break
                # end

                intStop += 1
            # end

            intArgs = int(objMatch.group(2))
            strArgs = strKernel[intStart:intStop].split(',')

            assert(intArgs == len(strArgs) - 1)

            strTensor = strArgs[0]
            intStrides = objVariables[strTensor].stride()

            strIndex = []

            for intArg in range(intArgs):
                strIndex.append('((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg] if torch.is_tensor(intStrides[intArg]) == False else intStrides[intArg].item()) + ')')
            # end

            strKernel = strKernel.replace('VALUE_' + str(intArgs) + '(' + strKernel[intStart:intStop] + ')', strTensor + '[' + str.join('+', strIndex) + ']')
        # end

        objCudacache[strKey] = {
            'strFunction': strFunction,
            'strKernel': strKernel
        }
    # end

    return strKey
# end


@cupy.memoize(for_each_device=True)
def cuda_launch(strKey:str):
    if 'CUDA_HOME' not in os.environ:
        os.environ['CUDA_HOME'] = cupy.cuda.get_cuda_path()
    # end

    return cupy.RawKernel(
        objCudacache[strKey]["strKernel"],
        objCudacache[strKey]["strFunction"],
        tuple(
            [
                "-I " + os.environ["CUDA_HOME"],
                "-I " + os.environ["CUDA_HOME"] + "/include",
            ]
        ),
    )
# end


##########################################################


def softsplat(tenIn:torch.Tensor, tenFlow:torch.Tensor, tenMetric:torch.Tensor, strMode:str):
    assert(strMode.split('-')[0] in ['sum', 'avg', 'linear', 'soft'])

    if strMode == 'sum': assert(tenMetric is None)
    if strMode == 'avg': assert(tenMetric is None)
    if strMode.split('-')[0] == 'linear': assert(tenMetric is not None)
    if strMode.split('-')[0] == 'soft': assert(tenMetric is not None)

    orig_dtype = tenIn.dtype
    tenIn = tenIn.float()
    tenFlow = tenFlow.float()
    tenMetric = tenMetric.float()

    if strMode == 'avg':
        tenIn = torch.cat([tenIn, tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]])], 1)

    elif strMode.split('-')[0] == 'linear':
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split('-')[0] == 'soft':
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    # end

    tenOut = softsplat_func.apply(tenIn, tenFlow)

    if strMode.split('-')[0] in ['avg', 'linear', 'soft']:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split('-')) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split('-')[1] == 'addeps':
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split('-')[1] == 'zeroeps':
            tenNormalize[tenNormalize == 0.0] = 1.0

        elif strMode.split('-')[1] == 'clipeps':
            tenNormalize = tenNormalize.clip(0.0000001, None)

        # end

        tenOut = tenOut[:, :-1, :, :] / tenNormalize
    # end

    return tenOut.to(orig_dtype)
# end


class softsplat_func(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, tenIn, tenFlow):
        tenOut = tenIn.new_zeros([tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]])

        if tenIn.is_cuda == True:
            cuda_launch(cuda_kernel('softsplat_out', '''
                extern "C" __global__ void __launch_bounds__(512) softsplat_out(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenFlow,
                    {{type}}* __restrict__ tenOut
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut) / SIZE_1(tenOut) ) % SIZE_0(tenOut);
                    const int intC = ( intIndex / SIZE_3(tenOut) / SIZE_2(tenOut)                  ) % SIZE_1(tenOut);
                    const int intY = ( intIndex / SIZE_3(tenOut)                                   ) % SIZE_2(tenOut);
                    const int intX = ( intIndex                                                    ) % SIZE_3(tenOut);

                    assert(SIZE_1(tenFlow) == 2);

                    {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
                    {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

                    if (isfinite(fltX) == false) { return; }
                    if (isfinite(fltY) == false) { return; }

                    {{type}} fltIn = VALUE_4(tenIn, intN, intC, intY, intX);

                    int intNorthwestX = (int) (floor(fltX));
                    int intNorthwestY = (int) (floor(fltY));
                    int intNortheastX = intNorthwestX + 1;
                    int intNortheastY = intNorthwestY;
                    int intSouthwestX = intNorthwestX;
                    int intSouthwestY = intNorthwestY + 1;
                    int intSoutheastX = intNorthwestX + 1;
                    int intSoutheastY = intNorthwestY + 1;

                    {{type}} fltNorthwest = (({{type}}) (intSoutheastX) - fltX) * (({{type}}) (intSoutheastY) - fltY);
                    {{type}} fltNortheast = (fltX - ({{type}}) (intSouthwestX)) * (({{type}}) (intSouthwestY) - fltY);
                    {{type}} fltSouthwest = (({{type}}) (intNortheastX) - fltX) * (fltY - ({{type}}) (intNortheastY));
                    {{type}} fltSoutheast = (fltX - ({{type}}) (intNorthwestX)) * (fltY - ({{type}}) (intNorthwestY));

                    if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(tenOut)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(tenOut))) {
                        atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intNorthwestY, intNorthwestX)], fltIn * fltNorthwest);
                    }

                    if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(tenOut)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(tenOut))) {
                        atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intNortheastY, intNortheastX)], fltIn * fltNortheast);
                    }

                    if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(tenOut)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(tenOut))) {
                        atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intSouthwestY, intSouthwestX)], fltIn * fltSouthwest);
                    }

                    if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(tenOut)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(tenOut))) {
                        atomicAdd(&tenOut[OFFSET_4(tenOut, intN, intC, intSoutheastY, intSoutheastX)], fltIn * fltSoutheast);
                    }
                } }
            ''', {
                'tenIn': tenIn,
                'tenFlow': tenFlow,
                'tenOut': tenOut
            }))(
                grid=tuple([int((tenOut.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenOut.nelement()), tenIn.data_ptr(), tenFlow.data_ptr(), tenOut.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )

        elif tenIn.is_cuda != True:
            assert(False)

        # end

        self.save_for_backward(tenIn, tenFlow)

        return tenOut
    # end

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(self, tenOutgrad):
        tenIn, tenFlow = self.saved_tensors

        tenOutgrad = tenOutgrad.contiguous(); assert(tenOutgrad.is_cuda == True)

        tenIngrad = tenIn.new_zeros([tenIn.shape[0], tenIn.shape[1], tenIn.shape[2], tenIn.shape[3]]) if self.needs_input_grad[0] == True else None
        tenFlowgrad = tenFlow.new_zeros([tenFlow.shape[0], tenFlow.shape[1], tenFlow.shape[2], tenFlow.shape[3]]) if self.needs_input_grad[1] == True else None

        if tenIngrad is not None:
            cuda_launch(cuda_kernel('softsplat_ingrad', '''
                extern "C" __global__ void __launch_bounds__(512) softsplat_ingrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenFlow,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenFlowgrad
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad) / SIZE_1(tenIngrad) ) % SIZE_0(tenIngrad);
                    const int intC = ( intIndex / SIZE_3(tenIngrad) / SIZE_2(tenIngrad)                     ) % SIZE_1(tenIngrad);
                    const int intY = ( intIndex / SIZE_3(tenIngrad)                                         ) % SIZE_2(tenIngrad);
                    const int intX = ( intIndex                                                             ) % SIZE_3(tenIngrad);

                    assert(SIZE_1(tenFlow) == 2);

                    {{type}} fltIngrad = 0.0f;

                    {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
                    {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

                    if (isfinite(fltX) == false) { return; }
                    if (isfinite(fltY) == false) { return; }

                    int intNorthwestX = (int) (floor(fltX));
                    int intNorthwestY = (int) (floor(fltY));
                    int intNortheastX = intNorthwestX + 1;
                    int intNortheastY = intNorthwestY;
                    int intSouthwestX = intNorthwestX;
                    int intSouthwestY = intNorthwestY + 1;
                    int intSoutheastX = intNorthwestX + 1;
                    int intSoutheastY = intNorthwestY + 1;

                    {{type}} fltNorthwest = (({{type}}) (intSoutheastX) - fltX) * (({{type}}) (intSoutheastY) - fltY);
                    {{type}} fltNortheast = (fltX - ({{type}}) (intSouthwestX)) * (({{type}}) (intSouthwestY) - fltY);
                    {{type}} fltSouthwest = (({{type}}) (intNortheastX) - fltX) * (fltY - ({{type}}) (intNortheastY));
                    {{type}} fltSoutheast = (fltX - ({{type}}) (intNorthwestX)) * (fltY - ({{type}}) (intNorthwestY));

                    if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(tenOutgrad)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(tenOutgrad))) {
                        fltIngrad += VALUE_4(tenOutgrad, intN, intC, intNorthwestY, intNorthwestX) * fltNorthwest;
                    }

                    if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(tenOutgrad)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(tenOutgrad))) {
                        fltIngrad += VALUE_4(tenOutgrad, intN, intC, intNortheastY, intNortheastX) * fltNortheast;
                    }

                    if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(tenOutgrad)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(tenOutgrad))) {
                        fltIngrad += VALUE_4(tenOutgrad, intN, intC, intSouthwestY, intSouthwestX) * fltSouthwest;
                    }

                    if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(tenOutgrad)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(tenOutgrad))) {
                        fltIngrad += VALUE_4(tenOutgrad, intN, intC, intSoutheastY, intSoutheastX) * fltSoutheast;
                    }

                    tenIngrad[intIndex] = fltIngrad;
                } }
            ''', {
                'tenIn': tenIn,
                'tenFlow': tenFlow,
                'tenOutgrad': tenOutgrad,
                'tenIngrad': tenIngrad,
                'tenFlowgrad': tenFlowgrad
            }))(
                grid=tuple([int((tenIngrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenIngrad.nelement()), tenIn.data_ptr(), tenFlow.data_ptr(), tenOutgrad.data_ptr(), tenIngrad.data_ptr(), None],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )
        # end

        if tenFlowgrad is not None:
            cuda_launch(cuda_kernel('softsplat_flowgrad', '''
                extern "C" __global__ void __launch_bounds__(512) softsplat_flowgrad(
                    const int n,
                    const {{type}}* __restrict__ tenIn,
                    const {{type}}* __restrict__ tenFlow,
                    const {{type}}* __restrict__ tenOutgrad,
                    {{type}}* __restrict__ tenIngrad,
                    {{type}}* __restrict__ tenFlowgrad
                ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
                    const int intN = ( intIndex / SIZE_3(tenFlowgrad) / SIZE_2(tenFlowgrad) / SIZE_1(tenFlowgrad) ) % SIZE_0(tenFlowgrad);
                    const int intC = ( intIndex / SIZE_3(tenFlowgrad) / SIZE_2(tenFlowgrad)                       ) % SIZE_1(tenFlowgrad);
                    const int intY = ( intIndex / SIZE_3(tenFlowgrad)                                             ) % SIZE_2(tenFlowgrad);
                    const int intX = ( intIndex                                                                   ) % SIZE_3(tenFlowgrad);

                    assert(SIZE_1(tenFlow) == 2);

                    {{type}} fltFlowgrad = 0.0f;

                    {{type}} fltX = ({{type}}) (intX) + VALUE_4(tenFlow, intN, 0, intY, intX);
                    {{type}} fltY = ({{type}}) (intY) + VALUE_4(tenFlow, intN, 1, intY, intX);

                    if (isfinite(fltX) == false) { return; }
                    if (isfinite(fltY) == false) { return; }

                    int intNorthwestX = (int) (floor(fltX));
                    int intNorthwestY = (int) (floor(fltY));
                    int intNortheastX = intNorthwestX + 1;
                    int intNortheastY = intNorthwestY;
                    int intSouthwestX = intNorthwestX;
                    int intSouthwestY = intNorthwestY + 1;
                    int intSoutheastX = intNorthwestX + 1;
                    int intSoutheastY = intNorthwestY + 1;

                    {{type}} fltNorthwest = 0.0f;
                    {{type}} fltNortheast = 0.0f;
                    {{type}} fltSouthwest = 0.0f;
                    {{type}} fltSoutheast = 0.0f;

                    if (intC == 0) {
                        fltNorthwest = (({{type}}) (-1.0f)) * (({{type}}) (intSoutheastY) - fltY);
                        fltNortheast = (({{type}}) (+1.0f)) * (({{type}}) (intSouthwestY) - fltY);
                        fltSouthwest = (({{type}}) (-1.0f)) * (fltY - ({{type}}) (intNortheastY));
                        fltSoutheast = (({{type}}) (+1.0f)) * (fltY - ({{type}}) (intNorthwestY));

                    } else if (intC == 1) {
                        fltNorthwest = (({{type}}) (intSoutheastX) - fltX) * (({{type}}) (-1.0f));
                        fltNortheast = (fltX - ({{type}}) (intSouthwestX)) * (({{type}}) (-1.0f));
                        fltSouthwest = (({{type}}) (intNortheastX) - fltX) * (({{type}}) (+1.0f));
                        fltSoutheast = (fltX - ({{type}}) (intNorthwestX)) * (({{type}}) (+1.0f));

                    }

                    for (int intChannel = 0; intChannel < SIZE_1(tenOutgrad); intChannel += 1) {
                        {{type}} fltIn = VALUE_4(tenIn, intN, intChannel, intY, intX);

                        if ((intNorthwestX >= 0) && (intNorthwestX < SIZE_3(tenOutgrad)) && (intNorthwestY >= 0) && (intNorthwestY < SIZE_2(tenOutgrad))) {
                            fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intNorthwestY, intNorthwestX) * fltIn * fltNorthwest;
                        }

                        if ((intNortheastX >= 0) && (intNortheastX < SIZE_3(tenOutgrad)) && (intNortheastY >= 0) && (intNortheastY < SIZE_2(tenOutgrad))) {
                            fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intNortheastY, intNortheastX) * fltIn * fltNortheast;
                        }

                        if ((intSouthwestX >= 0) && (intSouthwestX < SIZE_3(tenOutgrad)) && (intSouthwestY >= 0) && (intSouthwestY < SIZE_2(tenOutgrad))) {
                            fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intSouthwestY, intSouthwestX) * fltIn * fltSouthwest;
                        }

                        if ((intSoutheastX >= 0) && (intSoutheastX < SIZE_3(tenOutgrad)) && (intSoutheastY >= 0) && (intSoutheastY < SIZE_2(tenOutgrad))) {
                            fltFlowgrad += VALUE_4(tenOutgrad, intN, intChannel, intSoutheastY, intSoutheastX) * fltIn * fltSoutheast;
                        }
                    }

                    tenFlowgrad[intIndex] = fltFlowgrad;
                } }
            ''', {
                'tenIn': tenIn,
                'tenFlow': tenFlow,
                'tenOutgrad': tenOutgrad,
                'tenIngrad': tenIngrad,
                'tenFlowgrad': tenFlowgrad
            }))(
                grid=tuple([int((tenFlowgrad.nelement() + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[cuda_int32(tenFlowgrad.nelement()), tenIn.data_ptr(), tenFlow.data_ptr(), tenOutgrad.data_ptr(), None, tenFlowgrad.data_ptr()],
                stream=collections.namedtuple('Stream', 'ptr')(torch.cuda.current_stream().cuda_stream)
            )
        # end

        return tenIngrad, tenFlowgrad
    # end
# end
