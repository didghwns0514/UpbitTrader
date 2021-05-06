import asyncio

class Async_job:
	def __init__(self, _object, loop, args):
		self._object = _object if isinstance(_object, list) else [_object]
		self._loop = loop
		self._args = args
		# print(f'self._loop : {self._loop}')
		# print(f'self._args : {self._args}')

	async def make_async(self, _index):
		# async_container = [ await self.loop.run_in_executor(None, obj, *self._args[i]) \
		# 					for i, obj in enumerate(self._object) ]
		# print(f'self._object[_index] : {self._object[_index]}')
		# print(f'[self._args[_index] : {[self._args[_index]]}')
		#return await self._object[_index](*[self._args[_index]])
		return await self._loop.run_in_executor( None, self._object[_index], *self._args[_index] )

	async def __aenter__(self):
		async_container = [ self.make_async(i) for i in range(len(self._object)) ]
		return await asyncio.gather(*async_container, return_exceptions=True)

	async def __aexit__(self, exc_type, exc_value, traceback):
		pass


async def Async_main(_object, loop, args):
	async with Async_job(_object, loop, args) as result:
		return result


def async_wrapper(_object, args):
	"""

	:param _object:
	:param args: tuple wrapped in list
	:return:
	"""
	#loop = asyncio.get_event_loop()
	loop = asyncio.new_event_loop()
	tmp_rtn = loop.run_until_complete(Async_main(_object, loop, args))
	loop.close()

	return tmp_rtn

if __name__ == '__main__':
	import time
	cnt = 10
	def test_function(number1, number2):
		time.sleep(0.5)
		return f'you have waited for number : {number1}, {number2}\n'

	def test_function2(number1):
		time.sleep(0.5)
		return f'you have waited for number : {number1}\n'

	print('test arg : ',test_function(*[1, 2]))
	in_function = [test_function for i in range(cnt)]
	in_args = [[i, i] for i in range(cnt)]
	result = async_wrapper(in_function, in_args)
	print(f'result : {result}')


	in_function2 = [test_function2 for i in range(cnt)]
	in_args2 = [[i] for i in range(cnt)]
	result2 = async_wrapper(in_function2, in_args2)
	print(f'result : {result2}')


	print(2*24*60 // 200)
	print([i for i in range(2*24*60 //200)])
